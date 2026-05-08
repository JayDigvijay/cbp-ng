#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

#include <array>

using namespace hcm;


template<
    u64 LOGLB    = 6,   // 64B fetch block
    u64 NTABLES  = 10,  // number of tables (increased for MPP)
    // Ideally 5 hist tables => NTABLES = MPP_TABLES + 5 = 10
    u64 MPP_TABLES = 5, // number of MPP features (excluding base features)
    u64 MAXHIST  = 100, // maximum global history length, ideally 50
    u64 MINHIST  = 2,   // minimum global history length
    u64 WBITS    = 4,   // signed 4-bit weight (-8 to 7)
    u64 LOGTABLE = 13,  // 32KB hashed perceptron for P2
    
    // Novel features parameters
    u64 IMLI_BITS = 8,
    u64 MODULUS   = 4,
    u64 MODHIST_SIZE = 32,
    u64 RECENCY_DEPTH = 1,
    u64 ACYCLIC_SIZE = 16,
    u64 GSHARE_PRED_BANKS = 1,

    // gshareN_ahead parameters
    u64 LOGG = 19,
    u64 GHIST = 16,
    u64 N_gShare = 7
>
struct mpp_N_ahead : predictor {
    static_assert(LOGLB >= 2);
    static constexpr u64 LOGLINEINST = LOGLB - 2; // 4B instruction

    static constexpr u64 LINEINST = 1ull << LOGLINEINST;
    static constexpr u64 NUMHIST = NTABLES - MPP_TABLES; // number of tables using history, same as hashed perceptron

    static constexpr u64 YBITS  = WBITS + std::bit_width(NTABLES-1);
    static constexpr u64 TCBITS = 4; // corresponds to SPEED = 16
    static constexpr u64 THETABITS = YBITS + TCBITS;
    static constexpr u64 PATHBITS = 6;

    static_assert(LOGTABLE >= LOGLINEINST);
    static constexpr u64 index2_bits = LOGTABLE - LOGLINEINST;

    // gshareN_ahead constants
    static constexpr u64 LOGBANKS = std::bit_width(N_gShare);
    static constexpr u64 LOGLANES = std::bit_width(N_gShare - 1);
    static constexpr u64 LANES = 1 << LOGLANES;
    static constexpr u64 BANKS = 1 << LOGBANKS;
    static_assert(LOGG > (LOGLANES + LOGBANKS));
    static constexpr u64 index1_bits = LOGG - (LOGLANES + LOGBANKS);

    geometric_folds<NUMHIST, MINHIST, MAXHIST, index2_bits> gfolds;
    reg<1> true_block = 1;

    // ---- Novel Features State ----
    reg<IMLI_BITS> back_imli_counter = 0;
    
    //arr<reg<64>, RECENCY_DEPTH> recency_stack;
    //arr<reg<8>, RECENCY_DEPTH> recency_idx;
    reg<64> last_target_region = 0;
    
    reg<MODHIST_SIZE> mod_history = 0;
    reg<64> mod_path = 0;
    
    // Declared acyclic_path as array of regs
    arr<reg<1>, ACYCLIC_SIZE> acyclic_path;
    reg<64> blurry_path_history = 0;
    reg<64> current_line_pc = 0;

    // ---- for P2 (hashed perceptron) ----
    arr<reg<index2_bits>, NTABLES> index2;
    arr<reg<WBITS, i64>, NTABLES> readw[LINEINST];
    arr<reg<YBITS, i64>, LINEINST> yout;
    reg<LINEINST> p2;
    // dynamic update threshold
    reg<THETABITS, i64> theta_and_tc = 10 << TCBITS;

    // ---- for gshareN_ahead ----
    reg<GHIST> global_history_gShare;
    reg<index1_bits> index1[2];
    reg<LOGBANKS> path_gShare;
    reg<LOGBANKS> XB_gShare;
    reg<LANES> XL;
    arr<reg<LANES>, BANKS> block_pred_gShare[2];
    reg<LANES> unordered_pred_gShare;
    arr<reg<1>, LANES> pred_gShare;
    reg<1> true_block_gShare = 1;
    reg<1> last_condbr_dir_gShare = 1;
    reg<LOGLINEINST> block_entry;
    reg<N_gShare+1> rank_gShare;



    // gshareN_ahead RAMs
    ram<arr<val<LANES>, BANKS>, (1 << index1_bits)> ctr_hi_gShare;
    ram<val<1>, (BANKS << index1_bits)> ctr_lo_gShare[LANES];

    // ---- simulation artifacts ----
    u64 num_branch = 0;
    u64 block_size = 0;
    arr<reg<LOGLINEINST>, LINEINST> branch_offset;
    arr<reg<1>, LINEINST> branch_dir;

    // ---- RAMs ----
    ram<val<WBITS, i64>, (1 << index2_bits)> wtable[NTABLES][LINEINST] {{"P2 weight"}};

    val<1> line_end()
    {
        return (block_entry + block_size) == hard<LINEINST>{};
    }

    val<1> last_pred()
    {
        assert(num_branch <= N_gShare);
        return rank_gShare >> (N_gShare - num_branch);
    }

    void update_global_history(val<GHIST> injected_bits)
    {
        arr<val<1>, N_gShare + 1> num_cbr = (rank_gShare << num_branch).make_array(val<1>{});
        val<GHIST> shifted_ghist = arr<val<GHIST>, N_gShare + 1>{[&](int i) {
            return (global_history_gShare << std::max(i, 1)) & num_cbr[i].replicate(hard<GHIST>{}).concat();
        }}.fold_or();
        global_history_gShare = shifted_ghist ^ injected_bits;
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        inst_pc.fanout(hard<4>{});


        block_entry = select(true_block_gShare,
                             val<LOGLINEINST>{inst_pc >> 2},
                             val<LOGLINEINST>{block_entry + block_size});
        block_entry.fanout(hard<2>{});

        rank_gShare = select(true_block_gShare, val<N_gShare + 1>{1}, rank_gShare << num_branch);
        rank_gShare.fanout(hard<2>{});

        XL = select(true_block_gShare,
                    val<LOGLANES>{inst_pc >> 6}.decode().concat(),
                    XL.rotate_left(num_branch));
        XL.fanout(hard<LANES>{});

        execute_if(true_block_gShare, [&](){
            index1[1] = index1[0];
            if constexpr (GHIST <= index1_bits) {
                val<index1_bits> pc_bits = inst_pc >> (LOGBANKS + 2);
                index1[0] = pc_bits ^ (val<index1_bits>{global_history_gShare} << (index1_bits - GHIST));
            } else {
                index1[0] = global_history_gShare.make_array(val<index1_bits>{}).fold_xor();
            }
            block_pred_gShare[1] = block_pred_gShare[0];
            block_pred_gShare[0] = ctr_hi_gShare.read(index1[0]);
            path_gShare = XB_gShare + val<LOGBANKS>{num_branch} + ~last_condbr_dir_gShare;
            unordered_pred_gShare = block_pred_gShare[1].select(path_gShare);
            unordered_pred_gShare.fanout(hard<LANES>{});
        });

        XB_gShare = select(true_block_gShare,
                    val<LOGBANKS>{inst_pc >> 6},
                    val<LOGBANKS>{XB_gShare + val<LOGBANKS>{num_branch}});

        for (u64 i = 0; i < LANES; i++) {
            pred_gShare[i] = (unordered_pred_gShare & XL.rotate_left(i)) != hard<0>{};
        }
        pred_gShare.fanout(hard<2>{});
        block_size = 1;
        num_branch = 0;
        reuse_prediction(~line_end());
        return pred_gShare[num_branch];
    }

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        block_size++;
        reuse_prediction(~line_end());
        return pred_gShare[num_branch];
    }

    val<1> predict2(val<64> inst_pc)
    {   
        inst_pc.fanout(hard<2>{});
        val<index2_bits> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<NTABLES>{});
        mod_history.fanout(hard<2>{});
        mod_path.fanout(hard<2>{});
        gfolds.fanout(hard<2>{});
        
        // 1. Bimodal / Global History (Original Hashed Perceptron style for first 8 tables)
        arr<val<index2_bits>, NUMHIST> hashed_hist_indices = {[&](u64 i) {
            if (i == 0) {
                return lineaddr;
            } else {
                auto h = gfolds.template get<0>(i-1);
                return lineaddr ^ h;
            }
        }};
        
        // 2. IMLI (Inner-most Loop Iteration counter)

        // Back IMLI 
        val<index2_bits> back_imli_index = lineaddr ^ val<index2_bits>{back_imli_counter};
        
        // 3. MODHIST: Using mod_history directly because it is fanned out
        val<index2_bits> modhist_index = lineaddr ^ val<index2_bits>{mod_history};
        
        // 4. MODPATH: Using mod_path directly because it is fanned out
        val<index2_bits> modpath_index = lineaddr ^ val<index2_bits>{mod_path};
        
        // 5. GHISTMODPATH: Using mod_history and mod_path directly because they are fanned out
        val<index2_bits> ghistmodpath_index = lineaddr ^ val<index2_bits>{mod_history ^ mod_path};
        
        
        // 7. ACYCLIC: Used inst_pc directly and read from array acyclic_path
        // Compute the index as a combinational val
        val<std::bit_width(ACYCLIC_SIZE-1)> acyclic_idx = val<std::bit_width(ACYCLIC_SIZE-1)>{inst_pc % hard<ACYCLIC_SIZE>{}};
        // Statically fan out the array to read it combinationally
        acyclic_path.fanout(hard<2>{});
        // Select the element combinationally using the MUX select method
        val<1> acyclic_val = acyclic_path.select(acyclic_idx);

        val<index2_bits> acyclic_index = lineaddr ^ val<index2_bits>{acyclic_val};

        auto index2_vals = hashed_hist_indices
                            .append(back_imli_index)
                            .append(modhist_index)
                            .append(modpath_index)
                            .append(ghistmodpath_index)
                            //.append(blurrypath_index)
                            .append(acyclic_index);
                            //.append(global_history_index);
        index2 = index2_vals;
        index2.fanout(hard<2 * LINEINST>{});

        for (u64 i=0; i<NTABLES; i++) {
            auto dindex2 = index2[i].distribute(wtable[i]);
            for (u64 offset=0; offset<LINEINST; offset++) {
                readw[offset][i] = wtable[i][offset].read(dindex2[offset]);
            }
        }
        for (u64 offset=0; offset<LINEINST; offset++) {
            readw[offset].fanout(hard<2>{});
            yout[offset] = readw[offset].fold_add();
        }
        yout.fanout(hard<2>{});

        arr<val<1>, LINEINST> p2bits = [&](u64 offset) {
            auto [sign_bit, rest] = split<1, YBITS-1>(yout[offset]);
            return sign_bit;
        };
        p2 = p2bits.concat();
        p2.fanout(hard<LINEINST>{});

        val<1> taken = (block_entry & p2) != hard<0>{};

        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1)});
        return taken;
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        val<1> taken = ((block_entry << block_size) & p2) != hard<0>{};

        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1 - block_size)});
        block_size++;
        return taken;
    }

    void update_condbr(val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        assert(num_branch < LINEINST);
        branch_offset[num_branch] = branch_pc >> 2;
        branch_dir[num_branch] = taken;
        num_branch++;

        reuse_prediction(~(line_end() | last_pred()));
    }

    void update_cycle(instruction_info &block_end_info)
    {
        val<1> &mispredict = block_end_info.is_mispredict;
        val<64> &next_pc = block_end_info.next_pc;
        next_pc.fanout(hard<3>{});
        mispredict.fanout(hard<5>{});
        // Fanout fanned to LINEINST + 1 to cover both loop branches and block ending updates
        current_line_pc.fanout(hard<LINEINST+1>{});
        last_target_region.fanout(hard<LINEINST+1>{});
        back_imli_counter.fanout(hard<LINEINST+1>{});
        mod_history.fanout(hard<LINEINST+1>{});
        mod_path.fanout(hard<LINEINST+1>{});
        blurry_path_history.fanout(hard<LINEINST+1>{});

        if (num_branch == 0) {
            // gshareN_ahead updates
            update_global_history(next_pc >> 2);
            last_condbr_dir_gShare = 0;
            true_block_gShare = 1;


            val<1> line_end_mpp = block_entry >> (LINEINST - block_size);
            true_block.fanout(hard<2>{});
            val<1> actual_block = ~(true_block & line_end_mpp);
            actual_block.fanout(hard<MAXHIST+NUMHIST+3>{});
            execute_if(actual_block, [&](){
                next_pc.fanout(hard<2>{});
                gfolds.update(val<PATHBITS>{next_pc >> 2});
                true_block = 1;
            });
            return;
        }

        branch_dir.fanout(hard<2>{});
        branch_offset.fanout(hard<LINEINST + 1>{});
        //index1.fanout(hard<LINEINST*3>{});
        index2.fanout(hard<LINEINST>{});
        yout.fanout(hard<3>{});
        theta_and_tc.fanout(hard<2>{});
        p2.fanout(hard<2>{});

        u64 update_valid = (u64(1) << num_branch) - 1;
        arr<val<LINEINST>, LINEINST> update_mask = [&](u64 offset){
            arr<val<1>, LINEINST> match_offset = [&](u64 i){ return branch_offset[i] == offset; };
            return match_offset.concat() & update_valid;
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
        val<LINEINST> correct_mask = correct.concat();
        correct_mask.fanout(hard<3>{});

        auto [theta, tc] = split<YBITS, TCBITS>(theta_and_tc);
        theta.fanout(hard<LINEINST>{});
        arr<val<1>, LINEINST> weak = [&](u64 offset){
            auto absy = select(yout[offset] < hard<0>{}, -yout[offset], yout[offset]);
            return absy < theta;
        };
        val<LINEINST> weak_mask = weak.concat();
        weak_mask.fanout(hard<2>{});

        val<LINEINST> train_mask = branch_mask & (~correct_mask | weak_mask);
        train_mask.fanout(hard<2>{});
        arr<val<1>,LINEINST> train = train_mask.make_array(val<1>{});

        //val<LINEINST> disagree_mask = (p1 ^ p2) & branch_mask;
        //disagree_mask.fanout(hard<2>{});
        //arr<val<1>,LINEINST> disagree = disagree_mask.make_array(val<1>{});
        //disagree.fanout(hard<2>{});


        val<1> extra_cycle = train_mask != hard<0>{};
        need_extra_cycle(extra_cycle);

        val<LOGLINEINST> last_offset = branch_offset[num_branch-1];
        // Fanout fanned to LINEINST + 1: LINEINST for loop checks, 1 for block-ending update
        last_offset.fanout(hard<LINEINST+1>{});

        for (u64 offset=0; offset<LINEINST; offset++) {
            // Removed  because train inherits the fanout credit of train_mask
            execute_if(train[offset], [&](){
                for (u64 i=0; i<NTABLES; i++) {
                    wtable[i][offset].write(index2[i], update_ctr(readw[offset][i], ~branch_taken[offset]));
                }
            });
        }
        // ---- gshareN_ahead update_cycle logic ----
        static_assert(LANES <= 64);
        XL.fanout(hard<LANES + 1>{});
        index1[1].fanout(hard<2 * LANES + 1>{});
        path_gShare.fanout(hard<2 * LANES + BANKS>{});

        last_condbr_dir_gShare = branch_dir[num_branch - 1];
        last_condbr_dir_gShare.fanout(hard<LANES + 2>{});

        // access = mask telling which lanes are accessed by branches in the block
        arr<val<1>, LANES> access = arr<val<LANES>, LANES>{[&](u64 i) {
            return XL.rotate_left(i) & val<LANES>{-(i < num_branch)};
        }}.fold_or().make_array(val<1>{});

        // misp bank = bit vector pointing to the lane accessed by the mispredicted branch
        val<LANES> misp_bank = XL.rotate_left(num_branch - 1) & mispredict.replicate(hard<LANES>{}).concat();
        arr<val<1>, LANES> mispredicted = misp_bank.make_array(val<1>{});
        mispredicted.fanout(hard<2>{});

        // read hysteresis bit iff mispredict
        arr<val<1>, LANES> weak_gShare = [&](u64 i) {
            return execute_if(mispredicted[i], [&]() {
                return ~ctr_lo_gShare[i].read(concat(index1[1], path_gShare));
            });
        };

        // we need an extra cycle if there is a mispredict
        need_extra_cycle(mispredict);

        // update prediction if mispredict and the hysteresis bit is weak
        execute_if(mispredict, [&]() {
            arr<val<1>, LANES> stored_pred = unordered_pred_gShare.make_array(val<1>{});
            val<LANES> block_bundle = arr<val<1>, LANES>{
                [&](u64 i) {
                    return select(weak_gShare[i], last_condbr_dir_gShare, stored_pred[i]);
                }
            }.concat();
            block_bundle.fanout(hard<BANKS>{});
            arr<val<LANES>, BANKS> bundle = [&](u64 i) {
                return select(path_gShare == i, block_bundle, block_pred_gShare[1][i]);
            };
            ctr_hi_gShare.write(index1[1], bundle);
        });

        // update hysteresis
        for (u64 i = 0; i < LANES; i++) {
            execute_if(access[i], [&]() {
                ctr_lo_gShare[i].write(concat(index1[1], path_gShare), ~mispredicted[i]);
            });
        }

        // update the global history if this is a true block
        true_block_gShare = arr<val<1>, 4>{
            ~mispredict, last_condbr_dir_gShare, last_pred(), line_end()
        }.fold_or();
        true_block_gShare.fanout(hard<6>{});

        execute_if(true_block_gShare, [&]() {
            update_global_history(next_pc >> 2);
        });
        theta_and_tc = theta_and_tc - (branch_mask & weak_mask & correct_mask).ones() + (branch_mask & ~correct_mask).ones();

        val<1> line_end_mpp = block_entry >> (LINEINST - block_size);
        true_block = ~mispredict | branch_dir[num_branch-1] | line_end_mpp;
        true_block.fanout(hard<MAXHIST+NUMHIST+2>{});
        execute_if(true_block, [&](){
            gfolds.update(val<PATHBITS>{next_pc >> 2});

            // ---- Novel Speculative History updates are performed ONLY ONCE outside loop at block end ----
            // branch_pc for the block-ending conditional branch instruction
            val<64> branch_pc = current_line_pc | (val<64>{last_offset} << 2);
            val<1> taken = branch_dir[num_branch-1];
            branch_pc.fanout(hard<5>{});
            taken.fanout(hard<19>{});

            // 1. IMLI Loop Update (using array of regs lookup and write)
            val<1> is_forward = next_pc > branch_pc;

            execute_if(~is_forward, [&](){
                // Backward IMLI counter is incremented for taken backward branches, reset for not-taken backward branches
                back_imli_counter = select(taken, val<IMLI_BITS>{back_imli_counter + 1}, val<IMLI_BITS>{0});
            });

            // 2.& 3. MODHIST and MODPATH update
            val<1> mod_match = ((branch_pc % hard<MODULUS>{}) == 0);
            mod_match.fanout(hard<2>{});
            execute_if(mod_match, [&](){
                mod_history = (mod_history << 1) | taken;
                mod_path = (mod_path << 1) | val<64>{branch_pc >> 2};
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
