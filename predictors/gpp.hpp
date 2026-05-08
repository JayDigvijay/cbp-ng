#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

#include <array>

using namespace hcm;

/**
 * @brief GPP (Gainful Perceptron Predictor)
 * 
 * GPP is a highly optimized multi-feature branch predictor modeled as hardware logic 
 * using the HARCOM framework. It combines a fast, small gshare predictor (P1) with a 
 * more accurate, multi-feature hashed perceptron predictor (P2).
 * 
 * The design features complex state management (including geometric folds, loop iteration 
 * counters, modulo path/history, acyclic path tracking) to provide accurate predictions 
 * across varied workloads while observing strict credit-based fanout constraints for hardware model correctness.
 */
template<
    u64 LOGLB    = 6,   // 64B fetch block size (log2). Represents the size of instruction cache lines.
    u64 NTABLES  = 8,  // Number of perceptron weight tables in P2 (increased for multi-feature MPP).
    u64 MPP_TABLES = 3, // Number of novel MPP/GPP features (excluding base geometric history features).
    u64 MAXHIST  = 55, // Maximum global history length used in geometric history folding.
    u64 MINHIST  = 5,   // Minimum global history length used in geometric history folding.
    u64 WBITS    = 4,   // Signed 4-bit weight (-8 to 7) stored in the perceptron tables.
    u64 LOGTABLE = 13,  // Size (log2) of hashed perceptron weight tables (8192 entries per table).
    u64 LOGP1    = 15,  // Size (log2) of the gshare predictor table for P1 (32KB overall).
    u64 GHIST1   = 10,   // History length used for the fast P1 gshare predictor.
    
    // Novel GPP/MPP Feature Parameters
    u64 IMLI_BITS = 4,       // Bits for the Inner-most Loop Iteration (IMLI) counter.
    u64 MODULUS   = 4,       // Modulo parameter for the modulo path/history tracking.
    u64 MODHIST_SIZE = 32,   // History size for the modulo history register.
    u64 ACYCLIC_SIZE = 16   // Size of the acyclic path history register.
>
struct gpp : predictor {
    static_assert(LOGLB >= 2);
    
    // Constants derived from template parameters
    static constexpr u64 LOGLINEINST = LOGLB - 2; // Log2 number of instructions in a fetch block (4B per inst).
    static constexpr u64 LINEINST = 1ull << LOGLINEINST; // Total instructions per fetch block (e.g., 16 instructions).
    
    // Number of tables that use standard geometric history (e.g., 10 - 5 = 5 tables).
    static constexpr u64 NUMHIST = NTABLES - MPP_TABLES; 

    // Bit-width for the perceptron sum (yout) based on the weight size and table count.
    static constexpr u64 YBITS  = WBITS + std::bit_width(NTABLES-1);
    
    static constexpr u64 TCBITS = 4; // Hysteresis bits for the dynamic threshold tracker (equivalent to SPEED=16).
    static constexpr u64 THETABITS = YBITS + TCBITS; // Total bit-width for the dynamic threshold and training counter.
    static constexpr u64 PATHBITS = 6; // Number of PC bits used for geometric fold history hashing.

    static_assert(LOGP1 >= LOGLINEINST);
    static constexpr u64 index1_bits = LOGP1 - LOGLINEINST; // Index bits for each instruction offset's P1 table.
    static_assert(LOGTABLE >= LOGLINEINST);
    static constexpr u64 index2_bits = LOGTABLE - LOGLINEINST; // Index bits for each instruction offset's P2 table.

    // ---- Predictor Hardware State Declarations ----
    
    // Geometric history folds helper class: calculates and updates folded history bits 
    // for geometric path history matching across different table lengths.
    geometric_folds<NUMHIST, MINHIST, MAXHIST, index2_bits> gfolds;
    
    // Tracks whether the current instruction block's paths are non-speculative and valid.
    // Prevents updating global history on speculative/mispredicted execution branches.
    reg<1> true_block = 1;

    // ---- Novel Features State ----
    
    // Inner-most Loop Iteration (IMLI) backward loop counter: tracks the iterations of backward branches.
    reg<IMLI_BITS> back_imli_counter = 0;
    
    // Modulo History: tracks branch outcomes only for branches whose PCs match a specific modulo condition.
    reg<MODHIST_SIZE> mod_history = 0;
    
    // Modulo Path: tracks PC path history only for branches matching the modulo condition.
    reg<64> mod_path = 0;
    
    // Acyclic Path: records outcomes of branches mapping to a cyclic buffer of size ACYCLIC_SIZE.
    arr<reg<1>, ACYCLIC_SIZE> acyclic_path;
    
    // Current Line PC: base address of the current fetch block.
    reg<64> current_line_pc = 0;

    // ---- P1 (Fast GShare Predictor) State ----
    reg<GHIST1> global_history1;              // P1 global history register.
    reg<index1_bits> index1;                  // Hash index generated for accessing P1 tables.
    arr<reg<1>, LINEINST> readp1;             // Temporary register holding prediction bits read from table1_pred.
    reg<LINEINST> p1;                         // Vector of prediction bits for each instruction offset in the block.

    // ---- P2 (Hashed Perceptron Predictor) State ----
    arr<reg<index2_bits>, NTABLES> index2;    // Hashed indices generated for each of the NTABLES perceptron tables.
    arr<reg<WBITS, i64>, NTABLES> readw[LINEINST]; // Registers to hold weights read from wtable.
    arr<reg<YBITS, i64>, LINEINST> yout;      // Vector of accumulated perceptron sums for each instruction in the block.
    reg<LINEINST> p2;                         // Vector of prediction bits (signs of yout) for each instruction in the block.
    
    // Dynamic training threshold & training counter state register.
    reg<THETABITS, i64> theta_and_tc = 10 << TCBITS;

    // ---- Simulation Artifacts / Block Metadata ----
    u64 num_branch = 0;                       // Number of conditional branches processed in the current fetch block.
    u64 block_size = 0;                       // Fetch block length (how many instructions executed before block boundary).
    arr<reg<LOGLINEINST>, LINEINST> branch_offset; // Offsets within the block where branch instructions were located.
    arr<reg<1>, LINEINST> branch_dir;         // Directions (taken/not-taken) of branches processed.
    reg<LINEINST> block_entry;                // Mask representing which instructions in the fetch block have been executed.

    // ---- Random Access Memory (RAM) Tables ----
    
    // wtable: Perceptron weight tables. There are NTABLES distinct tables, each containing 
    // weights for every instruction offset within the LINEINST fetch block size.
    ram<val<WBITS, i64>, (1 << index2_bits)> wtable[NTABLES][LINEINST] {{"P2 weight"}};

    // table1_pred: Gshare prediction tables (1-bit predictor per index).
    ram<val<1>,(1 << index1_bits)> table1_pred[LINEINST] {"P1 pred"};
    
    // table1_hyst: Gshare hysteresis tables (1-bit hysteresis for 2-bit saturating counter logic).
    zone UPDATE_ONLY;
    ram<val<1>, (1 << index1_bits)> table1_hyst[LINEINST] {"P1 hyst"};

    /**
     * @brief Initializes tracking for a new fetch block.
     * 
     * Sets up block mask (block_entry), resets block size, and records current PC.
     * 
     * @param inst_pc The PC address of the first instruction starting the block.
     */
    void new_block(val<64> inst_pc)
    {
        // Fanout is set to 2: used combinationally for offset decoding and storing base PC.
        inst_pc.fanout(hard<2>{});
        val<LOGLINEINST> offset = inst_pc >> 2;
        
        // Decodes the starting offset into a bitmask (e.g., offset 3 becomes 00001000).
        // Concat translates the decoded bit list to a single register mask.
        block_entry = offset.fo1().decode().concat();
        
        // Block entry is fanned to hard<4*LINEINST> as it is read repeatedly by predictor and update loops.
        block_entry.fanout(hard<4*LINEINST>{});
        block_size = 1;
        current_line_pc = (inst_pc >> LOGLB) << LOGLB; // Align to fetch block boundary
    }

    /**
     * @brief Predicts branch directions for the entire fetch block using P1 (Gshare).
     * 
     * Computes the gshare index, reads table1_pred parallelly for all offsets, 
     * and returns whether the instruction at inst_pc is predicted taken.
     * 
     * @param inst_pc Address of the instruction making the prediction.
     * @return val<1> 1 if predicted taken, 0 if predicted not-taken.
     */
    val<1> predict1(val<64> inst_pc)
    {
        // Fanout fanned to 2: one for new_block init and one for indexing lineaddr.
        inst_pc.fanout(hard<2>{});
        new_block(inst_pc);

        val<std::max(index1_bits, GHIST1)> lineaddr = inst_pc >> LOGLB;
        global_history1.fanout(hard<2>{});
        
        // Generate Gshare index by XORing global history with fetch block line address.
        if constexpr (GHIST1 <= index1_bits) {
            index1 = lineaddr.fo1() ^ (val<index1_bits>{global_history1} << (index1_bits - GHIST1));
        } else {
            index1 = global_history1.make_array(val<index1_bits>{}).append(lineaddr.fo1()).fold_xor();
        }
        index1.fanout(hard<LINEINST+1>{});

        // Perform parallel reads from table1_pred for all potential instruction offsets in the block.
        for (u64 offset=0; offset<LINEINST; offset++) {
            readp1[offset] = table1_pred[offset].read(index1);
        }
        readp1.fanout(hard<2>{});
        p1 = readp1.concat();
        p1.fanout(hard<LINEINST>{});

        // Return the prediction corresponding to the starting instruction offset in the block.
        return (block_entry & p1) != hard<0>{};
    }

    /**
     * @brief Reuses the GShare (P1) prediction vector for subsequent instructions in the same fetch block.
     * 
     * Avoids accessing RAM tables multiple times in a single fetch cycle by shifting the block entry mask.
     * 
     * @param inst_pc Address of the instruction reusing the prediction.
     * @return val<1> Prediction bit.
     */
    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return ((block_entry << block_size) & p1) != hard<0>{};
    }

    /**
     * @brief Predicts branch direction using the multi-feature Hashed Perceptron (P2).
     * 
     * Computes hashed index indices for each table using geometric history folds, 
     * IMLI counter, modulo path/history, and acyclic path registers. 
     * Reads weights from wtable, sums them up, and returns prediction based on sign bit.
     * 
     * @param inst_pc Address of the instruction.
     * @return val<1> 1 if predicted taken, 0 if predicted not-taken.
     */
    val<1> predict2(val<64> inst_pc)
    {   
        inst_pc.fanout(hard<2>{});
        val<index2_bits> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<NTABLES + 5>{});
        mod_history.fanout(hard<2>{});
        mod_path.fanout(hard<2>{});
        gfolds.fanout(hard<2>{});
        
        // --- Feature 1: Hashed Geometric History Indices (First 5 tables) ---
        // Generates indices by XORing the line address with geometrically-sized global history folds.
        arr<val<index2_bits>, NUMHIST> hashed_hist_indices = {[&](u64 i) {
            if (i == 0) {
                return lineaddr; // Base table uses line address directly
            } else {
                auto h = gfolds.template get<0>(i-1);
                return lineaddr ^ h;
            }
        }};
        
        // --- Feature 2: Inner-most Loop Iteration (IMLI) Feature (Table 6) ---
        // Combines line address with backward branch iteration counts.
        val<index2_bits> back_imli_index = lineaddr ^ val<index2_bits>{back_imli_counter};
        
        // --- Feature 3: Modulo History + Path Combined Feature (Table 7) ---
        // Combines both modulo tracking features for multi-dimensional path correlation.
        val<index2_bits> ghistmodpath_index = lineaddr ^ val<index2_bits>{mod_history ^ mod_path};
        
        // --- Feature 4: Acyclic Path Feature (Table 8) ---
        // Combinationally selects acyclic path bit matching the branch PC index.
        val<std::bit_width(ACYCLIC_SIZE-1)> acyclic_idx = val<std::bit_width(ACYCLIC_SIZE-1)>{inst_pc % hard<ACYCLIC_SIZE>{}};
        acyclic_path.fanout(hard<2>{});
        val<1> acyclic_val = acyclic_path.select(acyclic_idx);
        val<index2_bits> acyclic_index = lineaddr ^ val<index2_bits>{acyclic_val};

        // Append all computed indices together to represent index2 for all NTABLES
        auto index2_vals = hashed_hist_indices
                            .append(back_imli_index)
                            .append(ghistmodpath_index)
                            .append(acyclic_index);
        index2 = index2_vals;
        index2.fanout(hard<2 * LINEINST>{});

        // Parallel weight reading: Retrieve weights from wtable for all tables and offsets
        for (u64 i=0; i<NTABLES; i++) {
            auto dindex2 = index2[i].distribute(wtable[i]);
            for (u64 offset=0; offset<LINEINST; offset++) {
                readw[offset][i] = wtable[i][offset].read(dindex2[offset].fo1());
            }
        }
        
        // Perceptron Summation: Accumulate (fold_add) weights for each instruction offset
        for (u64 offset=0; offset<LINEINST; offset++) {
            readw[offset].fanout(hard<2>{});
            yout[offset] = readw[offset].fold_add();
        }
        yout.fanout(hard<2>{});

        // Construct prediction vector p2 by extracting the sign bit (MSB) of yout
        arr<val<1>, LINEINST> p2bits = [&](u64 offset) {
            auto [sign_bit, rest] = split<1, YBITS-1>(yout[offset]);
            return sign_bit.fo1(); // MSB = 1 implies negative sum (not taken), MSB = 0 implies positive (taken)
        };
        p2 = p2bits.fo1().concat();
        p2.fanout(hard<LINEINST>{});

        val<1> taken = (block_entry & p2) != hard<0>{};

        // Pre-calculate reuse predictors for the remaining instruction offsets in the block
        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1)});
        return taken.fo1();
    }

    /**
     * @brief Reuses the Hashed Perceptron (P2) prediction vector for subsequent instructions in the block.
     * 
     * Shifts block entry masks and returns the pre-computed prediction bit from P2.
     * 
     * @param inst_pc Address of the instruction.
     * @return val<1> Prediction bit.
     */
    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        val<1> taken = ((block_entry << block_size) & (p2)) != hard<0>{};

        // Update reuse masks for instructions remaining after this one
        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1 - block_size)});
        block_size++;
        return taken.fo1();
    }

    /**
     * @brief Records branch outcomes processed within the current fetch block.
     * 
     * Updates block metadata lists to prepare for parallel update execution.
     * 
     * @param branch_pc Address of the branch instruction.
     * @param taken Direction of the branch.
     */
    void update_condbr(val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        assert(num_branch < LINEINST);
        branch_offset[num_branch] = branch_pc.fo1() >> 2;
        branch_dir[num_branch] = taken.fo1();
        num_branch++;
    }

    /**
     * @brief Executes the hardware update phase at the end of a fetch block cycle.
     * 
     * Computes parallel masks to train perceptron weights (upon misprediction or weak correct prediction), 
     * handles P1/P2 selection state updates (hysteresis tables), and updates speculative 
     * histories (geometric path folds, IMLI, modulo path, acyclic path) non-speculatively.
     * 
     * @param block_end_info Struct containing block outcome details (mispredict, next_pc, etc.)
     */
    void update_cycle(instruction_info &block_end_info)
    {
        val<1> &mispredict = block_end_info.is_mispredict;
        val<64> &next_pc = block_end_info.next_pc;
        next_pc.fanout(hard<3>{});
        
        // Static fanouts for end-of-block state update logic
        current_line_pc.fanout(hard<LINEINST+1>{});
        back_imli_counter.fanout(hard<LINEINST+1>{});
        mod_history.fanout(hard<LINEINST+1>{});
        mod_path.fanout(hard<LINEINST+1>{});

        // --- Case 1: No conditional branches in this fetch block ---
        if (num_branch == 0) {
            val<1> line_end = block_entry >> (LINEINST - block_size);
            true_block.fanout(hard<2>{});
            
            // If the path was correct (true_block) and block execution completes,
            // non-speculatively shift next_pc into global history registers.
            val<1> actual_block = ~(true_block & line_end.fo1());
            actual_block.fanout(hard<MAXHIST+NUMHIST+3>{});
            execute_if(actual_block, [&](){
                next_pc.fanout(hard<2>{});
                global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc >> 2};
                gfolds.update(val<PATHBITS>{next_pc >> 2});
                true_block = 1;
            });
            return;
        }

        // --- Case 2: Fetch block contained branches ---
        // Declare large fanouts for parallel update masks calculation
        branch_dir.fanout(hard<2>{});
        branch_offset.fanout(hard<LINEINST + 1>{});
        index1.fanout(hard<LINEINST*3>{});
        index2.fanout(hard<LINEINST>{});
        yout.fanout(hard<3>{});
        theta_and_tc.fanout(hard<2>{});
        p2.fanout(hard<2>{});

        u64 update_valid = (u64(1) << num_branch) - 1; // Mask of active branches in the block
        
        // Generate parallel mask matching instructions to their physical offset index
        arr<val<LINEINST>, LINEINST> update_mask = [&](u64 offset){
            arr<val<1>, LINEINST> match_offset = [&](u64 i){ return branch_offset[i] == offset; };
            return match_offset.fo1().concat() & update_valid;
        };
        update_mask.fanout(hard<2>{});

        // Branch presence vector: 1 if a branch occurred at a given instruction offset
        arr<val<1>, LINEINST> is_branch = [&](u64 offset){
            return update_mask[offset] != hard<0>{};
        };
        is_branch.fanout(hard<3>{});
        val<LINEINST> branch_mask = is_branch.concat();
        branch_mask.fanout(hard<5>{});

        // Branch taken direction vector for the fetch block offsets
        val<LINEINST> actualdirs = branch_dir.concat();
        actualdirs.fanout(hard<LINEINST>{});
        arr<val<1>, LINEINST> branch_taken = [&](u64 offset){
            return (actualdirs & update_mask[offset]) != hard<0>{};
        };
        branch_taken.fanout(hard<NTABLES+2>{});

        // Split perceptron prediction (P2) into per-offset bits
        auto p2_split = p2.make_array(val<1>{});
        p2_split.fanout(hard<3>{});
        arr<val<1>, LINEINST> correct = [&](u64 offset){
            return p2_split[offset] == branch_taken[offset];
        };
        val<LINEINST> correct_mask = correct.fo1().concat();
        correct_mask.fanout(hard<3>{});

        // Dynamic training threshold calculation: Check if perceptron sum was "weak" (near zero)
        auto [theta, tc] = split<YBITS, TCBITS>(theta_and_tc);
        theta.fanout(hard<LINEINST>{});
        arr<val<1>, LINEINST> weak = [&](u64 offset){
            // Absolute value of perceptron sum (yout)
            auto absy = select(yout[offset] < hard<0>{}, -yout[offset], yout[offset]);
            return absy.fo1() < theta; // Sum falls within training threshold
        };
        val<LINEINST> weak_mask = weak.fo1().concat();
        weak_mask.fanout(hard<2>{});

        // Perceptron training criteria: train if incorrect prediction OR correct but weak prediction
        val<LINEINST> train_mask = branch_mask & (~correct_mask | weak_mask);
        train_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> train = train_mask.make_array(val<1>{});

        // Check if P1 disagrees with a strong P2 prediction
        val<LINEINST> disagree_mask = (p1 ^ p2) & branch_mask & ~weak_mask;
        disagree_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> disagree = disagree_mask.make_array(val<1>{});
        disagree.fanout(hard<2>{});

        // Determine if Gshare (P1) is weak using hysteresis tables when predictors disagree
        arr<val<1>, LINEINST> p1_weak = [&](u64 offset) -> val<1> {
            return execute_if(disagree[offset], [&](){
                return ~table1_hyst[offset].read(index1);
            });
        };

        // Signal whether updates are needed this block cycle
        val<1> extra_cycle = (train_mask != hard<0>{}) | (disagree_mask != hard<0>{});
        need_extra_cycle(extra_cycle.fo1());

        // Parallel write back to gshare (P1) prediction tables upon disagreement on weak predictions
        for (u64 offset=0; offset<LINEINST; offset++) {
            execute_if(p1_weak[offset].fo1(), [&](){
                table1_pred[offset].write(index1, p2_split[offset]);
            });
        }
        val<LOGLINEINST> last_offset = branch_offset[num_branch-1];
        last_offset.fanout(hard<LINEINST+1>{});

        // Parallel update to gshare (P1) hysteresis tables
        for (u64 offset=0; offset<LINEINST; offset++) {
            execute_if(is_branch[offset], [&](){
                table1_hyst[offset].write(index1, ~disagree[offset]);
            });
        }

        // Parallel Perceptron Training: Increment/Decrement weights using saturating counters (update_ctr)
        for (u64 offset=0; offset<LINEINST; offset++) {
            execute_if(train[offset], [&](){
                for (u64 i=0; i<NTABLES; i++) {
                    wtable[i][offset].write(index2[i], update_ctr(readw[offset][i], ~branch_taken[offset]));
                }
            });
        }

        // Dynamic Threshold Adjustment: Adjust training threshold theta combinationally based on correctness/weakness ratios
        theta_and_tc = theta_and_tc - (branch_mask & weak_mask & correct_mask).ones() + (branch_mask & ~correct_mask).ones();

        // Determine if the last path outcome is final (non-speculative)
        val<1> line_end = block_entry >> (LINEINST - block_size);
        true_block = ~mispredict.fo1() | branch_dir[num_branch-1] | line_end.fo1();
        true_block.fanout(hard<MAXHIST+NUMHIST+2>{});
        execute_if(true_block, [&](){
            next_pc.fanout(hard<2>{});
            global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc >> 2};
            gfolds.update(val<PATHBITS>{next_pc >> 2});

            // --- Non-Speculative Multi-Feature History Updates (Performed ONCE at block-end boundary) ---
            val<64> branch_pc = current_line_pc.fo1() | (val<64>{last_offset} << 2);
            val<1> taken = branch_dir[num_branch-1];
            branch_pc.fanout(hard<5>{});
            taken.fanout(hard<4>{});

            // 1. IMLI Loop Counter Update:
            // Increment the iteration counter for backward taken branches; reset it for not-taken branches
            val<1> is_forward = next_pc > branch_pc;
            execute_if(~is_forward, [&](){
                back_imli_counter = select(taken, val<IMLI_BITS>{back_imli_counter.fo1() + 1}, val<IMLI_BITS>{0});
            });

            // 2. & 3. Modulo History and Path Updates:
            // Log outcome of branch and its PC
            val<1> mod_match = ((branch_pc % hard<MODULUS>{}) == 0);
            mod_match.fanout(hard<2>{});
            execute_if(mod_match, [&](){
                mod_history = (mod_history.fo1() << 1) | taken;
                mod_path = (mod_path.fo1() << 1) | val<64>{branch_pc >> 2};
            });

            // 4. Acyclic Path Updates:
            // Write outcomes into the cyclic buffer index matching the branch PC address remainder
            val<std::bit_width(ACYCLIC_SIZE-1)> update_acyclic_idx = val<std::bit_width(ACYCLIC_SIZE-1)>{branch_pc % hard<ACYCLIC_SIZE>{}};
            update_acyclic_idx.fanout(hard<ACYCLIC_SIZE>{});
            for (u64 i = 0; i < ACYCLIC_SIZE; i++) {
                execute_if(update_acyclic_idx == i, [&]() {
                    acyclic_path[i] = taken;
                });
            }
        });

        num_branch = 0; // Clear branches counter to prepare for next fetch block
    }
};
