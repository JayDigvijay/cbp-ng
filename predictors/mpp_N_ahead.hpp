#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

// This predictor uses ahead indexing:
//
//   [1] Seznec et al., "Multiple-block ahead branch predictors", ASPLOS 1996.
//   [2] Michaud et al., "Exploring instruction-fetch bandwidth requirement in
//       wide-issue superscalar processors", PACT 1999.
//   [3] Seznec & Fraboulet, "Effective ahead pipelining of instruction block
//       address generation", ISCA 2003.
//
// Instead of indexing block B1's prediction with the address of block B1, we index it
// with the address of the previous block, B0, and the path followed out of block B0.
// N is the maximum number of branches predicted per cycle (at most one taken branch).
// Unless block B0 ends on an indirect jump, there are at most N+1 paths out of B0.
// The N+1 possible block predictions are read simultaneously from multiple banks.
// Once the path is known in the next cycle, it is used to select one of the N+1 block predictions.
// As the N+1 paths out of B0 are not equally likely, and in order to use storage evenly,
// the bank associated with a given path depends on some bits XB of B0's address.
// Each of the N predictions for B1 is associated with a lane.
// To use lanes evenly, the lane depends on B1's address.


template<u64 LOGG=19, u64 GHIST=16, u64 N=7,
    u64 MPP_LOGLB    = 5,   // 32B fetch block
    u64 MPP_NTABLES  = 8,   // number of tables
    u64 MPP_MAXHIST  = 100, // maximum global history length
    u64 MPP_MINHIST  = 2,   // minimum global history length
    u64 MPP_WBITS    = 4,   // signed 4-bit weight (-8 to 7)
    u64 MPP_LOGTABLE = 13  // 32KB hashed perceptron for P2
>
struct mpp_N_ahead : predictor {
    // gshare with 2^LOGG entries, single prediction level (no overriding)
    // global history of GHIST bits
    // predicts up to N branches per cycle
    static constexpr u64 LOGBANKS = std::bit_width(N); // 3
    static constexpr u64 LOGLANES = std::bit_width(N-1);// 3
    static constexpr u64 LANES = 1 << LOGLANES;        // 8
    static constexpr u64 BANKS = 1 << LOGBANKS;        // 8
    static_assert(LOGG>(LOGLANES+LOGBANKS));
    static constexpr u64 index_bits = LOGG-(LOGLANES+LOGBANKS);
    // a block does not continue past a line boundary
    static constexpr u64 LOGLINEINST = 10;
    static constexpr u64 LINEINST = 1 << LOGLINEINST; // line size in instructions

    reg<GHIST> global_history;

    // pipelined over 2 cycles ([0]=current block, [1]=previous block)
    reg<index_bits> index[2];

    reg<LOGBANKS> path; // path out of previous block
    reg<LOGBANKS> XB; // for using banks evenly
    reg<LANES> XL; // for using lanes evenly

    arr<reg<LANES>,BANKS> block_pred[2]; // read block predictions
    reg<LANES> unordered_pred; // read prediction bits for the current block, unordered
    arr<reg<1>,LANES> pred; // read prediction bits for the current block, ordered
    reg<LANES> p1;

    // a true block is a block whose length is the same whether or not there is a mispredict
    reg<1> true_block = 1;
    reg<1> last_condbr_dir = 1;

    // for detecting the line boundary and the last available prediction
    reg<LOGLINEINST> block_entry; // offset of the entry point in the line
    reg<N+1> rank; // one-hot bit vector telling the rank of the current branch in the block

    // simulation artifacts, hardware cost not modeled accurately
    u64 num_branch = 0; // number of conditional branches in block so far
    u64 block_size = 0; // instructions in current block so far
    arr<reg<1>,N> branch_dir; // actual branch direction

    // RAMs
    ram<arr<val<LANES>,BANKS>,(1<<index_bits)> ctr_hi; // prediction bits
    ram<val<1>,(BANKS<<index_bits)> ctr_lo[LANES]; // hysteresis bit (0=weak, 1=strong)

    //////////// MPP components /////////////
    static_assert(MPP_LOGLB >= 2);
    static constexpr u64 MPP_LOGLINEINST = MPP_LOGLB - 2; // 4B instruction

    static constexpr u64 MPP_LINEINST = 1ull << MPP_LOGLINEINST;
    static_assert(MPP_LINEINST == LANES);
    static constexpr u64 MPP_NUMHIST = MPP_NTABLES - 1; // number of tables using history

    static constexpr u64 MPP_YBITS  = MPP_WBITS + std::bit_width(MPP_NTABLES-1);
    static constexpr u64 MPP_TCBITS = 4; // corresponds to SPEED = 16
    static constexpr u64 MPP_THETABITS = MPP_YBITS + MPP_TCBITS;
    static constexpr u64 MPP_PATHBITS = 6;

    static_assert(MPP_LOGTABLE >= MPP_LOGLINEINST);
    static constexpr u64 index2_bits = MPP_LOGTABLE - MPP_LOGLINEINST;

    geometric_folds<MPP_NUMHIST, MPP_MINHIST, MPP_MAXHIST, index2_bits> gfolds;

    arr<reg<index2_bits>, MPP_NTABLES> index2;
    arr<reg<MPP_WBITS, i64>, MPP_NTABLES> readw[MPP_LINEINST];
    arr<reg<MPP_YBITS, i64>, MPP_LINEINST> yout;
    reg<MPP_LINEINST> p2;
    // dynamic update threshold
    // packed counter representing (theta << TCBITS) + tc
    reg<MPP_THETABITS, i64> theta_and_tc = 10 << MPP_TCBITS;

    // ---- simulation artifacts ----
    arr<reg<MPP_LOGLINEINST>, MPP_LINEINST> branch_offset;

    // ---- RAMs ----
    // P2 weight tables
    ram<val<MPP_WBITS, i64>, (1 << index2_bits)> wtable[MPP_NTABLES][MPP_LINEINST] {{"P2 weight"}};

    //////////// END MPP components /////////////

    val<1> line_end()
    {
        return (block_entry + block_size) == hard<LINEINST>{};
    }

    val<1> last_pred()
    {
        assert(num_branch <= N);
        return rank >> (N-num_branch);
    }

    void update_global_history(val<GHIST> injected_bits)
    {
        // optimal global history length = tradeoff between footprint and branch correlations
        // footprint is a function of global history length in branches (not in blocks)
        arr<val<1>,N+1> num_cbr = (rank<<num_branch).make_array(val<1>{});
        val<GHIST> shifted_ghist = arr<val<GHIST>,N+1> {[&](int i){
            return (global_history<<std::max(i,1)) & num_cbr[i].replicate(hard<GHIST>{}).concat();
        }}.fold_or();
        global_history = shifted_ghist ^ injected_bits;
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        inst_pc.fanout(hard<4>{});
        true_block.fanout(hard<8+BANKS*2>{});

        // if the previous block was not a true block, we continue using the previous block predictions
        // (golden rule: never make a predictor's inputs depend on its outputs)

        block_entry = select(true_block,
                             val<LOGLINEINST>{inst_pc>>2},
                             val<LOGLINEINST>{block_entry+block_size});
        block_entry.fanout(hard<LINEINST+N+1>{});

        rank = select(true_block, val<N+1>{1}, rank<<num_branch);
        rank.fanout(hard<N+2>{});

        XL = select(true_block,
                    val<LOGLANES>{inst_pc>>6}.decode().concat(),
                    XL.rotate_left(num_branch));
        XL.fanout(hard<LANES>{});

        execute_if(true_block, [&](){
            index[1] = index[0];
            if constexpr (GHIST <= index_bits) {
                val<index_bits> pc_bits = inst_pc >> (LOGBANKS+2);
                index[0] = pc_bits ^ (val<index_bits>{global_history}<<(index_bits-GHIST));
            } else {
                index[0] = global_history.make_array(val<index_bits>{}).fold_xor();
            }
            block_pred[1] = block_pred[0];
            block_pred[0] = ctr_hi.read(index[0]);
            path = XB + num_branch + ~last_condbr_dir;
            unordered_pred = block_pred[1].select(path);
            unordered_pred.fanout(hard<LANES>{});
        });

        XB = select(true_block,
                    val<LOGBANKS>{inst_pc>>6},
                    val<LOGBANKS>{XB+num_branch});

        for (u64 i=0; i<LANES; i++) {
            pred[i] = (unordered_pred & XL.rotate_left(i)) != hard<0>{};
        }
        p1 = pred.concat();
        pred.fanout(hard<LINEINST*2>{});
        block_size = 1;
        num_branch = 0;
        reuse_prediction(~line_end());
        return pred[num_branch];
    };

    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        block_size++;
        reuse_prediction(~line_end());
        return pred[num_branch];
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
    val<index2_bits> lineaddr = inst_pc >> MPP_LOGLB;
        lineaddr.fanout(hard<MPP_NTABLES>{});
        gfolds.fanout(hard<2>{});

        // index2 = PC ^ folded_history
        for (u64 i=0; i<MPP_NTABLES; i++) {
            if (i == 0) {
                index2[i] = lineaddr;
            } else {
                index2[i] = lineaddr ^ gfolds.template get<0>(i-1);
            }
        }
        index2.fanout(hard<2>{});

        // read weights, then summed up
        for (u64 i=0; i<MPP_NTABLES; i++) {
            auto dindex2 = index2[i].distribute(wtable[i]);
            for (u64 offset=0; offset<MPP_LINEINST; offset++) {
                readw[offset][i] = wtable[i][offset].read(dindex2[offset]);
            }
        }
        for (u64 offset=0; offset<MPP_LINEINST; offset++) {
            readw[offset].fanout(hard<2>{});
            yout[offset] = readw[offset].fold_add();
        }
        yout.fanout(hard<2>{});

        // per-offset predictions
        arr<val<1>, MPP_LINEINST> p2bits = [&](u64 offset) {
            auto [sign_bit, rest] = split<1, MPP_YBITS-1>(yout[offset]);
            return sign_bit;
        };
        p2 = p2bits.concat();
        p2.fanout(hard<MPP_LINEINST>{});

        // prediction for the first instruction
        val<1> taken = (block_entry & p2) != hard<0>{};

        // determine block termination
        //reuse_prediction(~line_end());
        return taken;
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        // prediction for subsequent instructions
        val<1> taken = ((block_entry << block_size) & p2) != hard<0>{};

        // determine block termination
        //reuse_prediction(~line_end());
        block_size++;
        return taken;
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        assert(num_branch<N);
        branch_dir[num_branch] = taken;
        
        /////// MPP Part /////////
        branch_offset[num_branch] = branch_pc >> 2;
        num_branch++;
        /////////////////////////

        reuse_prediction(~(line_end() | last_pred()));
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info)
    {
        val<1> &mispredict = block_end_info.is_mispredict;
        val<64> &next_pc = block_end_info.next_pc;
        global_history.fanout(hard<N+1>{});

        if (num_branch == 0) {
            // no conditional branch in this block
            next_pc.fanout(hard<2>{});
            update_global_history(next_pc>>2);
            last_condbr_dir = 0;
            true_block = 1;
            gfolds.update(val<MPP_PATHBITS>{next_pc >> 2});
            return; // stop here
        }

        /////// GSHARE Calculations //////////////
        static_assert(LANES<=64);
        XL.fanout(hard<LANES+1>{});
        index[1].fanout(hard<2*LANES+1>{});
        mispredict.fanout(hard<LANES+2>{});
        path.fanout(hard<2*LANES+BANKS>{});

        last_condbr_dir = branch_dir[num_branch-1];
        last_condbr_dir.fanout(hard<LANES+2>{});

        // misp bank = bit vector pointing to the lane accessed by the mispredicted branch
        // (all zero if no mispredict)
        val<LANES> misp_bank = XL.rotate_left(num_branch-1) & mispredict.replicate(hard<LANES>{}).concat();
        arr<val<1>,LANES> mispredicted = misp_bank.make_array(val<1>{});
        mispredicted.fanout(hard<2>{});

        /////////// End GSHARE Calculations /////////

        /////// MPP Calculations /////////
        branch_dir.fanout(hard<2>{});
        branch_offset.fanout(hard<MPP_LINEINST>{});
        index2.fanout(hard<MPP_LINEINST>{});
        yout.fanout(hard<3>{});
        theta_and_tc.fanout(hard<2>{});
        p2.fanout(hard<2>{});

        // masks mapping executed branches to offset
        u64 update_valid = (u64(1) << num_branch) - 1;
        arr<val<MPP_LINEINST>, MPP_LINEINST> update_mask = [&](u64 offset){
            arr<val<1>, MPP_LINEINST> match_offset = [&](u64 i){ return branch_offset[i] == offset; };
            return match_offset.concat() & update_valid;
        };
        update_mask.fanout(hard<2>{});

        // Is there a branch instruction at the offset?
        arr<val<1>, MPP_LINEINST> is_branch = [&](u64 offset){
            return update_mask[offset] != hard<0>{};
        };
        is_branch.fanout(hard<3>{});
        val<MPP_LINEINST> branch_mask = is_branch.concat();
        branch_mask.fanout(hard<5>{});

        // is the outcome of the branch at the offset taken?
        val<MPP_LINEINST> actualdirs = branch_dir.concat();
        actualdirs.fanout(hard<MPP_LINEINST>{});
        arr<val<1>, MPP_LINEINST> branch_taken = [&](u64 offset){
            return (actualdirs & update_mask[offset]) != hard<0>{};
        };
        branch_taken.fanout(hard<MPP_NTABLES+1>{});

        // is the P2 (=final) prediction correct?
        auto p2_split = p2.make_array(val<1>{});
        p2_split.fanout(hard<3>{});
        arr<val<1>, MPP_LINEINST> correct = [&](u64 offset){
            return p2_split[offset] == branch_taken[offset];
        };
        val<MPP_LINEINST> correct_mask = correct.concat();
        correct_mask.fanout(hard<3>{});

        // is the prediction weak?
        auto [theta, tc] = split<MPP_YBITS, MPP_TCBITS>(theta_and_tc);
        theta.fanout(hard<MPP_LINEINST>{});
        arr<val<1>, MPP_LINEINST> mpp_weak = [&](u64 offset){
            auto absy = select(yout[offset] < hard<0>{}, -yout[offset], yout[offset]);
            return absy < theta;
        };
        val<MPP_LINEINST> mpp_weak_mask = mpp_weak.concat();
        mpp_weak_mask.fanout(hard<2>{});

        // perceptrons are updated if the prediction is wrong or weak
        val<MPP_LINEINST> mpp_train_mask = branch_mask & (~correct_mask | mpp_weak_mask);
        mpp_train_mask.fanout(hard<2>{});
        arr<val<1>,MPP_LINEINST> train = mpp_train_mask.make_array(val<1>{});

        // did P1 and P2 disagree?
        val<LANES> disagree_mask = (p1 ^ p2) & branch_mask;
        disagree_mask.fanout(hard<2>{});
        arr<val<1>,LANES> disagree = disagree_mask.make_array(val<1>{});
        disagree.fanout(hard<2>{});

        // read P1 hysteresis if P1 and P2 disagree
        arr<val<1>,LANES> gshare_weak = [&](u64 i){
            return execute_if(disagree[i], [&](){
                return ~ctr_lo[i].read(concat(index[1],path));
            });
        };
        /////////// End MPP Calculations /////////

        // we need an extra cycle if there is a mispredict
        val<1> extra_cycle = (mpp_train_mask != hard<0>{}) | (disagree_mask != hard<0>{});
        need_extra_cycle(extra_cycle);

        //////////// Begin gShare Update /////////////
        // update prediction if mispredict and the hysteresis bit is weak
        execute_if(disagree_mask != hard<0>{}, [&](){
            arr<val<1>,LANES> stored_pred = unordered_pred.make_array(val<1>{});
            val<LANES> block_bundle = arr<val<1>,LANES>{
                [&](u64 i){
                    return select(gshare_weak[i], last_condbr_dir, stored_pred[i]);
                }
            }.concat();
            block_bundle.fanout(hard<BANKS>{});
            arr<val<LANES>,BANKS> bundle = [&](u64 i){
                return select(path==i, block_bundle, block_pred[1][i]);
            };
            ctr_hi.write(index[1],bundle);
        });

        // update hysteresis
        for (u64 i=0; i<LANES; i++) {
            execute_if(is_branch[i], [&](){
                ctr_lo[i].write(concat(index[1],path),~disagree[i]);
            });
        }

        /////////// End gShare Update ////////////

        //////////// Begin MPP Update ////////////
        
        // ---- P2 (hashed perceptron) update ----
        // weight update
        for (u64 offset=0; offset<MPP_LINEINST; offset++) {
            execute_if(train[offset], [&](){
                for (u64 i=0; i<MPP_NTABLES; i++) {
                    wtable[i][offset].write(index2[i], update_ctr(readw[offset][i], ~branch_taken[offset]));
                }
            });
        }

        // update packed counter
        // decrement on weak & correct
        // increment on mispredict
        theta_and_tc = theta_and_tc - (branch_mask & mpp_weak_mask & correct_mask).ones() + (branch_mask & ~correct_mask).ones();

        num_branch = 0;
        
        /////////// End MPP Update ////////////

        ////// True Block Update ///////////
        true_block = arr<val<1>,4> {
            ~mispredict, last_condbr_dir, last_pred(), line_end()
        }.fold_or();
        true_block.fanout(hard<MPP_MAXHIST+MPP_NUMHIST+2>{});
        // update the global history if this is a true block
        execute_if(true_block, [&](){
            update_global_history(next_pc>>2);
            gfolds.update(val<MPP_PATHBITS>{next_pc >> 2});
        });

    }
};
