#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

#include <array>

using namespace hcm;


template<
    u64 LOGLB    = 6,   // 64B fetch block
    u64 NTABLES  = 10,  // number of tables (increased for MPP)
    u64 MPP_TABLES = 5, // number of MPP features (excluding base features)
    u64 MAXHIST  = 50, // maximum global history length
    u64 MINHIST  = 2,   // minimum global history length
    u64 WBITS    = 4,   // signed 4-bit weight (-8 to 7)
    u64 LOGTABLE = 13,  // 32KB hashed perceptron for P2
    u64 LOGP1    = 15,  // 4KB gshare for P1
    u64 GHIST1   = 11,   // P1 gshare history length
    
    // Novel features parameters
    u64 IMLI_BITS = 8,
    u64 MODULUS   = 4,
    u64 MODHIST_SIZE = 32,
    u64 RECENCY_DEPTH = 1,
    u64 BLURRY_SHIFT = 4,
    u64 ACYCLIC_SIZE = 16
>
struct mpp : predictor {
    static_assert(LOGLB >= 2);
    static constexpr u64 LOGLINEINST = LOGLB - 2; // 4B instruction

    static constexpr u64 LINEINST = 1ull << LOGLINEINST;
    static constexpr u64 NUMHIST = NTABLES - MPP_TABLES; // number of tables using history, same as hashed perceptron

    static constexpr u64 YBITS  = WBITS + std::bit_width(NTABLES-1);
    static constexpr u64 TCBITS = 4; // corresponds to SPEED = 16
    static constexpr u64 THETABITS = YBITS + TCBITS;
    static constexpr u64 PATHBITS = 6;

    static_assert(LOGP1 >= LOGLINEINST);
    static constexpr u64 index1_bits = LOGP1 - LOGLINEINST;
    static_assert(LOGTABLE >= LOGLINEINST);
    static constexpr u64 index2_bits = LOGTABLE - LOGLINEINST;

    geometric_folds<NUMHIST, MINHIST, MAXHIST, index2_bits> gfolds;
    reg<1> true_block = 1;

    // ---- Novel Features State ----
    reg<IMLI_BITS> imli_counter = 0;
    
    //arr<reg<64>, RECENCY_DEPTH> recency_stack;
    //arr<reg<8>, RECENCY_DEPTH> recency_idx;
    reg<64> last_region = 0;
    
    reg<MODHIST_SIZE> mod_history = 0;
    reg<64> mod_path = 0;
    
    // Declared acyclic_path as array of regs
    arr<reg<1>, ACYCLIC_SIZE> acyclic_path;
    reg<64> blurry_path_history = 0;
    reg<64> current_line_pc = 0;

    // ---- for P1 (gshare) ----
    reg<GHIST1> global_history1;
    reg<index1_bits> index1;
    arr<reg<1>, LINEINST> readp1;
    reg<LINEINST> p1;

    // ---- for P2 (hashed perceptron) ----
    arr<reg<index2_bits>, NTABLES> index2;
    arr<reg<WBITS, i64>, NTABLES> readw[LINEINST];
    arr<reg<YBITS, i64>, LINEINST> yout;
    reg<LINEINST> p2;
    // dynamic update threshold
    reg<THETABITS, i64> theta_and_tc = 10 << TCBITS;

    // ---- simulation artifacts ----
    u64 num_branch = 0;
    u64 block_size = 0;
    arr<reg<LOGLINEINST>, LINEINST> branch_offset;
    arr<reg<1>, LINEINST> branch_dir;
    reg<LINEINST> block_entry;

    // ---- RAMs ----
    ram<val<WBITS, i64>, (1 << index2_bits)> wtable[NTABLES][LINEINST] {{"P2 weight"}};

    ram<val<1>, (1 << index1_bits)> table1_pred[LINEINST] {"P1 pred"};
    zone UPDATE_ONLY;
    ram<val<1>, (1 << index1_bits)> table1_hyst[LINEINST] {"P1 hyst"};

    void new_block(val<64> inst_pc)
    {
        // Removed .fo1() because inst_pc was fanned out inside predict1
        val<LOGLINEINST> offset = inst_pc >> 2;
        block_entry = offset.fo1().decode().concat();
        block_entry.fanout(hard<4*LINEINST>{});
        block_size = 1;
        // inst_pc is fanned out and can be read a second time here without .fo1()
        current_line_pc = (inst_pc >> LOGLB) << LOGLB;
    }

    val<1> predict1(val<64> inst_pc)
    {
        // Fanout is fanned to 3: 2 for new_block and 1 for predict1 lineaddr
        inst_pc.fanout(hard<3>{});
        new_block(inst_pc);

        // Using inst_pc directly because it is fanned out
        val<std::max(index1_bits, GHIST1)> lineaddr = inst_pc >> LOGLB;
        global_history1.fanout(hard<2>{});
        if constexpr (GHIST1 <= index1_bits) {
            index1 = lineaddr.fo1() ^ (val<index1_bits>{global_history1} << (index1_bits - GHIST1));
        } else {
            index1 = global_history1.make_array(val<index1_bits>{}).append(lineaddr.fo1()).fold_xor();
        }
        index1.fanout(hard<LINEINST+1>{});

        for (u64 offset=0; offset<LINEINST; offset++) {
            readp1[offset] = table1_pred[offset].read(index1);
        }
        readp1.fanout(hard<2>{});
        p1 = readp1.concat();
        p1.fanout(hard<LINEINST>{});

        return (block_entry & p1) != hard<0>{};
    }

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return ((block_entry << block_size) & p1) != hard<0>{};
    }

    val<1> predict2(val<64> inst_pc)
    {   
        // Used inst_pc directly instead of .fo1() because it is already fanned out
        inst_pc.fanout(hard<2>{});
        val<index2_bits> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<NTABLES>{});
        mod_history.fanout(hard<2>{});
        mod_path.fanout(hard<2>{});
        gfolds.fanout(hard<2>{});
        
        // 1. Bimodal / Global History (Original Hashed Perceptron style for first 8 tables)
        arr<val<index2_bits>, NUMHIST> hashed_hist_indices = {[&](u64 i) {
            if (i == 0) {
                return lineaddr.fo1();
            } else {
                auto h = gfolds.template get<0>(i-1);
                return lineaddr.fo1() ^ h;
            }
        }};
        
        // 2. IMLI (Inner-most Loop Iteration counter)
        val<index2_bits> imli_index = lineaddr ^ val<index2_bits>{imli_counter};
        
        // 3. MODHIST: Using mod_history directly because it is fanned out
        val<index2_bits> modhist_index = lineaddr ^ val<index2_bits>{mod_history};
        
        // 4. MODPATH: Using mod_path directly because it is fanned out
        val<index2_bits> modpath_index = lineaddr ^ val<index2_bits>{mod_path};
        
        // 5. GHISTMODPATH: Using mod_history and mod_path directly because they are fanned out
        val<index2_bits> ghistmodpath_index = lineaddr ^ val<index2_bits>{mod_history ^ mod_path};
        
        // 6. BLURRYPATH
       // val<index2_bits> blurrypath_index = lineaddr ^ val<index2_bits>{blurry_path_history};
        
        // 7. ACYCLIC: Used inst_pc directly without .fo1() and read from array acyclic_path
        // Compute the index as a combinational val
        val<std::bit_width(ACYCLIC_SIZE-1)> acyclic_idx = val<std::bit_width(ACYCLIC_SIZE-1)>{inst_pc % hard<ACYCLIC_SIZE>{}};
        // Statically fan out the array to read it combinationally
        acyclic_path.fanout(hard<2>{});
        // Select the element combinationally using the MUX select method
        val<1> acyclic_val = acyclic_path.select(acyclic_idx);

        val<index2_bits> acyclic_index = lineaddr ^ val<index2_bits>{acyclic_val};

        // 8. GLOBAL HISTORY: Reuse gShare history
        /*
        global_history1.fanout(hard<2>{});
        val<index2_bits> global_history_index = [&](){
            if constexpr (GHIST1 <= index2_bits) {
                return lineaddr.fo1() ^ (val<index2_bits>{global_history1} << (index2_bits - GHIST1));
            } else {
                return global_history1.make_array(val<index2_bits>{}).append(lineaddr.fo1()).fold_xor();
            }
        }();
        */

        auto index2_vals = hashed_hist_indices
                            .append(imli_index)
                            .append(modhist_index)
                            .append(modpath_index)
                            .append(ghistmodpath_index)
                            //.append(blurrypath_index)
                            .append(acyclic_index)
                            //.append(global_history_index);
        index2 = index2_vals;
        index2.fanout(hard<2 * LINEINST>{});

        for (u64 i=0; i<NTABLES; i++) {
            auto dindex2 = index2[i].distribute(wtable[i]);
            for (u64 offset=0; offset<LINEINST; offset++) {
                readw[offset][i] = wtable[i][offset].read(dindex2[offset].fo1());
            }
        }
        for (u64 offset=0; offset<LINEINST; offset++) {
            readw[offset].fanout(hard<2>{});
            yout[offset] = readw[offset].fold_add();
        }
        yout.fanout(hard<2>{});

        arr<val<1>, LINEINST> p2bits = [&](u64 offset) {
            auto [sign_bit, rest] = split<1, YBITS-1>(yout[offset]);
            return sign_bit.fo1();
        };
        p2 = p2bits.fo1().concat();
        p2.fanout(hard<LINEINST>{});

        val<1> taken = (block_entry & p2) != hard<0>{};

        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1)});
        return taken.fo1();
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        val<1> taken = ((block_entry << block_size) & p2) != hard<0>{};

        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1 - block_size)});
        block_size++;
        return taken.fo1();
    }

    void update_condbr(val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        assert(num_branch < LINEINST);
        branch_offset[num_branch] = branch_pc.fo1() >> 2;
        branch_dir[num_branch] = taken.fo1();
        num_branch++;
    }

    void update_cycle(instruction_info &block_end_info)
    {
        val<1> &mispredict = block_end_info.is_mispredict;
        val<64> &next_pc = block_end_info.next_pc;
        next_pc.fanout(hard<3>{});
        // Fanout fanned to LINEINST + 1 to cover both loop branches and block ending updates
        current_line_pc.fanout(hard<LINEINST+1>{});
        last_region.fanout(hard<LINEINST+1>{});
        imli_counter.fanout(hard<LINEINST+1>{});
        mod_history.fanout(hard<LINEINST+1>{});
        mod_path.fanout(hard<LINEINST+1>{});
        blurry_path_history.fanout(hard<LINEINST+1>{});

        if (num_branch == 0) {
            val<1> line_end = block_entry >> (LINEINST - block_size);
            true_block.fanout(hard<2>{});
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

        branch_dir.fanout(hard<2>{});
        branch_offset.fanout(hard<LINEINST + 1>{});
        index1.fanout(hard<LINEINST*3>{});
        index2.fanout(hard<LINEINST>{});
        yout.fanout(hard<3>{});
        theta_and_tc.fanout(hard<2>{});
        p2.fanout(hard<2>{});

        u64 update_valid = (u64(1) << num_branch) - 1;
        arr<val<LINEINST>, LINEINST> update_mask = [&](u64 offset){
            arr<val<1>, LINEINST> match_offset = [&](u64 i){ return branch_offset[i] == offset; };
            return match_offset.fo1().concat() & update_valid;
        };
        update_mask.fanout(hard<2>{});

        arr<val<1>, LINEINST> is_branch = [&](u64 offset){
            return update_mask[offset] != hard<0>{};
        };
        is_branch.fanout(hard<3>{});
        val<LINEINST> branch_mask = is_branch.concat();
        branch_mask.fanout(hard<5>{});

        val<LINEINST> actualdirs = branch_dir.concat();
        actualdirs.fanout(hard<LINEINST>{});
        arr<val<1>, LINEINST> branch_taken = [&](u64 offset){
            return (actualdirs & update_mask[offset]) != hard<0>{};
        };
        branch_taken.fanout(hard<NTABLES+2>{});

        auto p2_split = p2.make_array(val<1>{});
        p2_split.fanout(hard<3>{});
        arr<val<1>, LINEINST> correct = [&](u64 offset){
            return p2_split[offset] == branch_taken[offset];
        };
        val<LINEINST> correct_mask = correct.fo1().concat();
        correct_mask.fanout(hard<3>{});

        auto [theta, tc] = split<YBITS, TCBITS>(theta_and_tc);
        theta.fanout(hard<LINEINST>{});
        arr<val<1>, LINEINST> weak = [&](u64 offset){
            auto absy = select(yout[offset] < hard<0>{}, -yout[offset], yout[offset]);
            return absy.fo1() < theta;
        };
        val<LINEINST> weak_mask = weak.fo1().concat();
        weak_mask.fanout(hard<2>{});

        val<LINEINST> train_mask = branch_mask & (~correct_mask | weak_mask);
        train_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> train = train_mask.make_array(val<1>{});

        val<LINEINST> disagree_mask = (p1 ^ p2) & branch_mask;
        disagree_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> disagree = disagree_mask.make_array(val<1>{});
        disagree.fanout(hard<2>{});

        arr<val<1>, LINEINST> p1_weak = [&](u64 offset) -> val<1> {
            return execute_if(disagree[offset], [&](){
                return ~table1_hyst[offset].read(index1);
            });
        };

        val<1> extra_cycle = (train_mask != hard<0>{}) | (disagree_mask != hard<0>{});
        need_extra_cycle(extra_cycle.fo1());

        for (u64 offset=0; offset<LINEINST; offset++) {
            execute_if(p1_weak[offset].fo1(), [&](){
                table1_pred[offset].write(index1, p2_split[offset]);
            });
        }
        val<LOGLINEINST> last_offset = branch_offset[num_branch-1];
        // Fanout fanned to LINEINST + 1: LINEINST for loop checks, 1 for block-ending update
        last_offset.fanout(hard<LINEINST+1>{});

        for (u64 offset=0; offset<LINEINST; offset++) {
            execute_if(is_branch[offset], [&](){
                // Statically, each offset branch updates its own offset table1_hyst RAM, which is perfectly parallel
                table1_hyst[offset].write(index1, ~disagree[offset]);
            });
        }

        for (u64 offset=0; offset<LINEINST; offset++) {
            // Removed .fo1() because train inherits the fanout credit of train_mask
            execute_if(train[offset], [&](){
                for (u64 i=0; i<NTABLES; i++) {
                    wtable[i][offset].write(index2[i], update_ctr(readw[offset][i], ~branch_taken[offset]));
                }
            });
        }

        theta_and_tc = theta_and_tc - (branch_mask & weak_mask & correct_mask).ones() + (branch_mask & ~correct_mask).ones();

        val<1> line_end = block_entry >> (LINEINST - block_size);
        true_block = ~mispredict.fo1() | branch_dir[num_branch-1] | line_end.fo1();
        true_block.fanout(hard<MAXHIST+NUMHIST+2>{});
        execute_if(true_block, [&](){
            next_pc.fanout(hard<2>{});
            global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc >> 2};
            gfolds.update(val<PATHBITS>{next_pc >> 2});

            // ---- Novel Speculative History updates are performed ONLY ONCE outside loop at block end ----
            // branch_pc for the block-ending conditional branch instruction
            val<64> branch_pc = current_line_pc.fo1() | (val<64>{last_offset} << 2);
            val<1> taken = branch_dir[num_branch-1];
            branch_pc.fanout(hard<5>{});
            taken.fanout(hard<4>{});

            // 1. IMLI Loop Update (using array of regs lookup and write)
            val<1> is_forward = next_pc > branch_pc;

            execute_if(is_forward.fo1(), [&](){
                imli_counter = select(taken, val<IMLI_BITS>{0}, val<IMLI_BITS>{imli_counter.fo1() + 1});
            });

            // 2.& 3. MODHIST and MODPATH update
            val<1> mod_match = ((branch_pc % hard<MODULUS>{}) == 0);
            mod_match.fanout(hard<2>{});
            execute_if(mod_match, [&](){
                mod_history = (mod_history.fo1() << 1) | taken;
                mod_path = (mod_path.fo1() << 1) | val<64>{branch_pc >> 2};
            });


            // 5. BLURRYPATH update
            val<64> current_region = branch_pc >> BLURRY_SHIFT;
            current_region.fanout(hard<2>{});
            val<64> lr = last_region.fo1();
            lr.fanout(hard<2>{});
            execute_if(current_region != lr, [&](){
                blurry_path_history = (blurry_path_history.fo1() << 1) | val<64>{lr};
                last_region = current_region;
            });

            // 6. ACYCLIC update: Write to acyclic_path array
            val<std::bit_width(ACYCLIC_SIZE-1)> update_acyclic_idx = val<std::bit_width(ACYCLIC_SIZE-1)>{branch_pc % hard<ACYCLIC_SIZE>{}};
            update_acyclic_idx.fanout(hard<ACYCLIC_SIZE>{});
            for (u64 i = 0; i < ACYCLIC_SIZE; i++) {
                execute_if(update_acyclic_idx == i, [&]() {
                    acyclic_path[i] = taken;
                });
            }
        });

        num_branch = 0;
    }
};
