#include "../cbp.hpp"
#include "../harcom.hpp"
#include "common.hpp"

#include <array>

using namespace hcm;

template<
    u64 LOGLB = 6,
    u64 NUMG = 16,        // Number of global tables for TAGE
    u64 LOGG = 11,        // Log size of each tagged table
    u64 LOGB = 12,        // Log size of bimodal table
    u64 TAGW = 11,        // Tag width
    u64 GHIST = 1000,     // History length for TAGE (extended to 1000 as in CBP2025)
    u64 LOGP1 = 14,       // Log size of gshare (P1)
    u64 GHIST1 = 6,       // Gshare history length
    
    // SC parameters
    u64 LOGBIAS = 11,     // Logsize of SC tables
    u64 PERCWIDTH = 6,    // Statistical Corrector counter width
    u64 IMLI_BITS = 10,   // Iteration iteration counter bits
    u64 RECENCY_DEPTH = 8,
    u64 BLURRY_SHIFT = 4,
    u64 ACYCLIC_SIZE = 16
>
struct tage_sc_l : predictor {
    static_assert(LOGLB > 2);
    static constexpr u64 LOGLINEINST = LOGLB - 2;
    static constexpr u64 LINEINST = 1 << LOGLINEINST;
    static_assert(LOGP1 > LOGLINEINST);
    static_assert(LOGB > LOGLINEINST);
    static constexpr u64 index1_bits = LOGP1 - LOGLINEINST;
    static constexpr u64 bindex_bits = LOGB - LOGLINEINST;
    static_assert(TAGW > LOGLINEINST);
    static constexpr u64 HTAGBITS = TAGW - LOGLINEINST;
    static constexpr u64 PATHBITS = 6;

    geometric_folds<NUMG, 3, GHIST, LOGG, HTAGBITS> gfolds;
    reg<1> true_block = 1;

    // P1 Gshare
    reg<GHIST1> global_history1;
    reg<index1_bits> index1;
    arr<reg<1>, LINEINST> readp1;
    reg<LINEINST> p1;

    // P2 TAGE
    reg<bindex_bits> bindex;
    arr<reg<LOGG>, NUMG> gindex;
    arr<reg<HTAGBITS>, NUMG> htag;

    arr<reg<1>, LINEINST> readb;
    arr<reg<TAGW>, NUMG> readt;
    arr<reg<1>, NUMG> readc;
    arr<reg<2>, NUMG> readh;
    arr<reg<1>, NUMG> readu;
    reg<NUMG> notumask;

    arr<reg<NUMG+1>, LINEINST> match;
    arr<reg<NUMG+1>, LINEINST> match1;
    arr<reg<NUMG+1>, LINEINST> match2;

    arr<reg<1>, LINEINST> pred1;
    arr<reg<1>, LINEINST> pred2;
    reg<LINEINST> p2;

    // Statistical Corrector (SC) table parameters
    static constexpr u64 GNB = 5;
    static constexpr u64 ANB = 4;
    static constexpr u64 BNB = 1;
    static constexpr u64 FNB = 1;
    static constexpr u64 PNB = 1;

    static constexpr u64 LNB = 4;
    static constexpr u64 SNB = 4;

    // --- SC Corrector Tables (Banked per offset for conflict-free parallel execution) ---
    rwram<PERCWIDTH, (1 << LOGBIAS), 4> bias_pc[LINEINST] {{"sc_bias_pc"}};
    rwram<PERCWIDTH, (1 << LOGBIAS), 4> bias_pclmap[LINEINST] {{"sc_bias_pclmap"}};
    rwram<PERCWIDTH, 128, 4> bias_alt[LINEINST] {{"sc_bias_alt"}};
    rwram<PERCWIDTH, 64, 4> bias_bim[LINEINST] {{"sc_bias_bim"}};
    rwram<PERCWIDTH, 512, 4> bias_lmhcalt[LINEINST] {{"sc_bias_lmhcalt"}};

    // --- IMLI Tables ---
    rwram<PERCWIDTH, 1024, 4> ibias[LINEINST] {{"sc_ibias"}};
    rwram<PERCWIDTH, 1024, 4> iibias[LINEINST] {{"sc_iibias"}};
    rwram<PERCWIDTH, 1024, 4> isbias[LINEINST] {{"sc_isbias"}};
    rwram<PERCWIDTH, 1024, 4> iisbias[LINEINST] {{"sc_iisbias"}};

    // --- History Correlation Tables ---
    rwram<PERCWIDTH, 2048, 4> ggehl[GNB][LINEINST] {{"sc_ggehl"}};
    rwram<PERCWIDTH, 2048, 4> agehl[ANB][LINEINST] {{"sc_agehl"}};
    rwram<PERCWIDTH, 2048, 4> bgehl[BNB][LINEINST] {{"sc_bgehl"}};
    rwram<PERCWIDTH, 2048, 4> fgehl[FNB][LINEINST] {{"sc_fgehl"}};
    rwram<PERCWIDTH, 1024, 4> pgehl[PNB][LINEINST] {{"sc_pgehl"}};

    // --- Local Branch Tables ---
    rwram<PERCWIDTH, 2048, 4> lgehl[LNB][LINEINST] {{"sc_lgehl"}};
    rwram<PERCWIDTH, 2048, 4> sgehl[SNB][LINEINST] {{"sc_sgehl"}};

    // --- History registers for corrector ---
    reg<IMLI_BITS> imli_counter = 0;
    reg<64> blurry_path_history = 0;
    reg<64> current_line_pc = 0;
    reg<64> last_region = 0;
    reg<64> mod_history = 0;
    reg<64> mod_path = 0;

    // --- TAGE RAM arrays ---
    ram<val<1>, (1 << index1_bits)> table1_pred[LINEINST] {"P1 pred"};
    ram<val<TAGW>, (1 << LOGG)> gtag[NUMG] {"tags"};
    ram<val<1>, (1 << LOGG)> gpred[NUMG] {"gpred"};
    rwram<2, (1 << LOGG), 4> ghyst[NUMG] {"ghyst"};
    rwram<1, (1 << LOGG), 4> ubit[NUMG] {"u"};
    ram<val<1>, (1 << bindex_bits)> bim[LINEINST] {"bpred"};
    zone UPDATE_ONLY;
    ram<val<1>, (1 << index1_bits)> table1_hyst[LINEINST] {"P1 hyst"};
    ram<val<1>, (1 << bindex_bits)> bhyst[LINEINST] {"bhyst"};

    // Simulation trackers
    u64 num_branch = 0;
    u64 block_size = 0;
    arr<reg<LOGLINEINST>, LINEINST> branch_offset;
    arr<reg<1>, LINEINST> branch_dir;
    reg<LINEINST> block_entry;

    void new_block(val<64> inst_pc)
    {
        // Removed .fo1() because inst_pc is fanned out inside predict1
        val<LOGLINEINST> offset = inst_pc >> 2;
        block_entry = offset.fo1().decode().concat();
        block_entry.fanout(hard<6*LINEINST>{});
        block_size = 1;
        // inst_pc is fanned out and read a second time here safely without .fo1()
        current_line_pc = (inst_pc >> LOGLB) << LOGLB;
    }

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // Fanout fanned to 3: 2 for new_block reads and 1 for predict1 lineaddr
        inst_pc.fanout(hard<3>{});
        new_block(inst_pc);
        
        // Using inst_pc directly because it is fanned out
        val<std::max(index1_bits, GHIST1)> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<2>{});
        if constexpr (GHIST1 <= index1_bits) {
            index1 = lineaddr ^ (val<index1_bits>{global_history1} << (index1_bits - GHIST1));
        } else {
            index1 = global_history1.make_array(val<index1_bits>{}).append(lineaddr).fold_xor();
        }
        index1.fanout(hard<LINEINST>{});
        for (u64 offset = 0; offset < LINEINST; offset++) {
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
        val<std::max(bindex_bits, LOGG)> lineaddr = inst_pc >> LOGLB;
        lineaddr.fanout(hard<1 + NUMG * 2>{});
        gfolds.fanout(hard<2>{});

        bindex = lineaddr;
        bindex.fanout(hard<LINEINST>{});
        for (u64 i = 0; i < NUMG; i++) {
            gindex[i] = lineaddr ^ gfolds.template get<0>(i);
        }
        gindex.fanout(hard<4>{});

        for (u64 i = 0; i < NUMG; i++) {
            htag[i] = val<HTAGBITS>{lineaddr}.reverse() ^ gfolds.template get<1>(i);
        }
        htag.fanout(hard<2>{});

        for (u64 offset = 0; offset < LINEINST; offset++) {
            readb[offset] = bim[offset].read(bindex);
        }
        readb.fanout(hard<2>{});
        for (u64 i = 0; i < NUMG; i++) {
            readt[i] = gtag[i].read(gindex[i]);
            readc[i] = gpred[i].read(gindex[i]);
            readh[i] = ghyst[i].read(gindex[i]);
            readu[i] = ubit[i].read(gindex[i]);
        }
        readt.fanout(hard<LINEINST + 1>{});
        readc.fanout(hard<3>{});
        readh.fanout(hard<2>{});
        readu.fanout(hard<2>{});
        notumask = ~readu.concat();
        notumask.fanout(hard<2>{});

        val<NUMG> gpreds = readc.concat();
        gpreds.fanout(hard<LINEINST>{});
        arr<val<NUMG+1>, LINEINST> preds = [&](u64 offset) { return concat(readb[offset], gpreds); };
        preds.fanout(hard<2 * LINEINST>{});

        arr<val<1>, NUMG> htagcmp_split = [&](int i) { return val<HTAGBITS>{readt[i]} == htag[i]; };
        val<NUMG> htagcmp = htagcmp_split.fo1().concat();
        htagcmp.fanout(hard<LINEINST>{});

        static_loop<LINEINST>([&]<u64 offset>() {
            arr<val<1>, NUMG> tagcmp = [&](int i) { return val<LOGLINEINST>{readt[i] >> HTAGBITS} == hard<offset>{}; };
            match[offset] = concat(val<1>{1}, tagcmp.fo1().concat() & htagcmp);
        });
        match.fanout(hard<2>{});

        for (u64 offset = 0; offset < LINEINST; offset++) {
            match1[offset] = match[offset].one_hot();
        }
        match1.fanout(hard<3>{});
        for (u64 offset = 0; offset < LINEINST; offset++) {
            pred1[offset] = (match1[offset] & preds[offset]) != hard<0>{};
        }
        pred1.fanout(hard<2>{});

        for (u64 offset = 0; offset < LINEINST; offset++) {
            match2[offset] = (match[offset] ^ match1[offset]).one_hot();
        }
        match2.fanout(hard<2>{});
        for (u64 offset = 0; offset < LINEINST; offset++) {
            pred2[offset] = (match2[offset] & preds[offset]) != hard<0>{};
        }
        pred2.fanout(hard<2>{});

        // --- Base TAGE predictions ---
        p2 = pred1.concat();
        p2.fanout(hard<LINEINST>{});
        val<1> taken = (block_entry & p2) != hard<0>{};
        taken.fanout(hard<2>{});
        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1)});
        return taken;
    }

    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        val<1> taken = ((block_entry << block_size) & p2) != hard<0>{};
        taken.fanout(hard<2>{});
        reuse_prediction(~val<1>{block_entry >> (LINEINST - 1 - block_size)});
        block_size++;
        return taken;
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

        if (num_branch == 0) {
            val<1> line_end = block_entry >> (LINEINST - block_size);
            true_block.fanout(hard<2>{});
            val<1> actual_block = ~(true_block & line_end.fo1());
            actual_block.fanout(hard<GHIST + NUMG * 2 + 2>{});
            execute_if(actual_block, [&]() {
                next_pc.fanout(hard<2>{});
                global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc >> 2};
                gfolds.update(val<PATHBITS>{next_pc >> 2});
                true_block = 1;
            });
            return;
        }

        mispredict.fanout(hard<NUMG + 2>{});
        val<1> correct_pred = ~mispredict;
        correct_pred.fanout(hard<NUMG + 2>{});
        index1.fanout(hard<LINEINST * 3>{});
        p2.fanout(hard<2>{});
        bindex.fanout(hard<LINEINST * 3>{});
        gindex.fanout(hard<4>{});
        htag.fanout(hard<3>{});
        readb.fanout(hard<2>{});
        readt.fanout(hard<4>{});
        readc.fanout(hard<2>{});
        match1.fanout(hard<3>{});
        match2.fanout(hard<2>{});
        pred1.fanout(hard<2>{});
        pred2.fanout(hard<2 + NUMG>{});
        branch_offset.fanout(hard<LINEINST + NUMG + 1>{});
        branch_dir.fanout(hard<2>{});
        gfolds.fanout(hard<2>{});

        val<LOGLINEINST> last_offset = branch_offset[num_branch - 1];
        last_offset.fanout(hard<4 * NUMG + 2>{});

        u64 update_valid = (u64(1) << num_branch) - 1;
        arr<val<LINEINST>, LINEINST> update_mask = [&](u64 offset) {
            arr<val<1>, LINEINST> match_offset = [&](u64 i) { return branch_offset[i] == offset; };
            return match_offset.fo1().concat() & update_valid;
        };
        update_mask.fanout(hard<2>{});

        arr<val<1>, LINEINST> is_branch = [&](u64 offset) {
            return update_mask[offset] != hard<0>{};
        };
        is_branch.fanout(hard<6>{});
        val<LINEINST> branch_mask = is_branch.concat();

        val<LINEINST> actualdirs = branch_dir.concat();
        actualdirs.fanout(hard<LINEINST>{});

        arr<val<1>, LINEINST> branch_taken = [&](u64 offset) {
            return (actualdirs & update_mask[offset]) != hard<0>{};
        };
        branch_taken.fanout(hard<3>{});

        arr<val<NUMG + 1>, LINEINST> actual_match1 = [&](u64 offset) {
            return select(is_branch[offset], match1[offset], val<NUMG + 1>{0});
        };
        actual_match1.fanout(hard<2>{});

        val<NUMG> primary_mask = actual_match1.fold_or();
        primary_mask.fanout(hard<2>{});
        arr<val<1>, NUMG> primary = primary_mask.make_array(val<1>{});
        primary.fanout(hard<3>{});

        arr<val<1>, LINEINST> primary_wrong = [&](u64 offset) {
            return pred1[offset] != branch_taken[offset];
        };
        primary_wrong.fanout(hard<2>{});

        val<NUMG> mispmask = mispredict.replicate(hard<NUMG>{}).concat();
        arr<val<1>, NUMG> last_tagcmp = [&](int i) { return readt[i] == concat(last_offset, htag[i]); };
        val<NUMG + 1> last_match1 = last_tagcmp.fo1().append(1).concat().one_hot();
        last_match1.fanout(hard<2>{});
        val<NUMG> postmask = mispmask.fo1() & val<NUMG>(last_match1 - 1);
        postmask.fanout(hard<2>{});
        val<NUMG> candallocmask = postmask & notumask;
        candallocmask.fanout(hard<2>{});
        val<NUMG> collamask = candallocmask.reverse();
        collamask.fanout(hard<2>{});
        val<NUMG> collamask1 = collamask.one_hot();
        collamask1.fanout(hard<3>{});
        val<NUMG> collamask2 = (collamask ^ collamask1).one_hot();
        val<NUMG> collamask12 = select(val<2>{std::rand()} == hard<0>{}, collamask2.fo1(), collamask1);
        arr<val<1>, NUMG> allocate = collamask12.fo1().reverse().make_array(val<1>{});
        allocate.fanout(hard<7>{});

        arr<val<1>, NUMG> bdir = [&](u64 i) {
            val<LOGLINEINST> tag_offset = readt[i] >> HTAGBITS;
            val<LOGLINEINST> offset = select(allocate[i], last_offset, tag_offset.fo1());
            offset.fanout(hard<LINEINST>{});
            arr<val<1>, LINEINST> match_offset = [&](u64 j) { return branch_offset[j] == offset; };
            return (match_offset.fo1().concat() & update_valid & actualdirs) != hard<0>{};
        };
        bdir.fanout(hard<2>{});

        arr<val<1>, NUMG> badpred1 = [&](u64 i) {
            return readc[i] != bdir[i];
        };
        badpred1.fanout(hard<3>{});

        arr<val<1>, NUMG> altdiffer = [&](u64 i) {
            val<LOGLINEINST> tag_offset = readt[i] >> HTAGBITS;
            return readc[i] != pred2.select(tag_offset.fo1());
        };

        arr<val<1>, NUMG> goodpred = [&](u64 i) {
            val<LOGLINEINST> tag_offset = readt[i] >> HTAGBITS;
            return (tag_offset.fo1() != last_offset) | correct_pred;
        };

        val<LINEINST> disagree_mask = (p1 ^ p2) & branch_mask.fo1();
        disagree_mask.fanout(hard<2>{});
        arr<val<1>, LINEINST> disagree = disagree_mask.make_array(val<1>{});
        disagree.fanout(hard<2>{});

        arr<val<1>, LINEINST> p1_weak = [&](u64 offset) -> val<1> {
            return execute_if(disagree[offset], [&]() {
                return ~table1_hyst[offset].read(index1);
            });
        };

        arr<val<1>, LINEINST> b_weak = [&](u64 offset) -> val<1> {
            val<1> bim_primary = actual_match1[offset] >> NUMG;
            return execute_if(bim_primary.fo1() & primary_wrong[offset], [&]() {
                return ~bhyst[offset].read(bindex);
            });
        };

        arr<val<1>, NUMG> g_weak = [&](u64 i) -> val<1> {
            return primary[i] & badpred1[i] & (readh[i] == hard<0>{});
        };

        val<1> some_badpred1 = (primary_mask & badpred1.concat()) != hard<0>{};
        val<1> extra_cycle = some_badpred1.fo1() | mispredict | (disagree_mask != hard<0>{});
        extra_cycle.fanout(hard<NUMG * 2 + 1>{});
        need_extra_cycle(extra_cycle);

        for (u64 i = 0; i < NUMG; i++) {
            execute_if(allocate[i], [&]() { gtag[i].write(gindex[i], concat(last_offset, htag[i])); });
        }

        arr<val<1>, NUMG> update_u = [&](u64 i) {
            return primary[i] & altdiffer[i].fo1();
        };
        val<1> noalloc = (candallocmask == hard<0>{});
        val<NUMG> uclearmask = postmask & noalloc.fo1().replicate(hard<NUMG>{}).concat();
        arr<val<1>, NUMG> uclear = uclearmask.fo1().make_array(val<1>{});
        uclear.fanout(hard<2>{});
        for (u64 i = 0; i < NUMG; i++) {
            execute_if(update_u[i].fo1() | allocate[i] | uclear[i], [&]() {
                val<1> newu = goodpred[i].fo1() & ~allocate[i] & ~uclear[i];
                ubit[i].write(gindex[i], newu.fo1(), extra_cycle);
            });
        }

        auto p2_split = p2.make_array(val<1>{});
        for (u64 offset = 0; offset < LINEINST; offset++) {
            execute_if(p1_weak[offset].fo1(), [&]() {
                // Removed .fo1() because p2_split[offset] is fanned out LINEINST times
                table1_pred[offset].write(index1, p2_split[offset]);
            });
        }
        for (u64 offset = 0; offset < LINEINST; offset++) {
            execute_if(is_branch[offset], [&]() {
                table1_hyst[offset].write(index1, ~disagree[offset]);
            });
        }

        for (u64 offset = 0; offset < LINEINST; offset++) {
            execute_if(b_weak[offset].fo1(), [&]() {
                bim[offset].write(bindex, branch_taken[offset]);
            });
        }
        for (u64 offset = 0; offset < LINEINST; offset++) {
            val<1> bim_primary = match1[offset] >> NUMG;
            execute_if(is_branch[offset] & bim_primary.fo1(), [&]() {
                bhyst[offset].write(bindex, ~primary_wrong[offset]);
            });
        }

        for (u64 i = 0; i < NUMG; i++) {
            execute_if(g_weak[i].fo1() | allocate[i], [&]() {
                gpred[i].write(gindex[i], bdir[i]);
            });
        }
        for (u64 i = 0; i < NUMG; i++) {
            execute_if(primary[i] | allocate[i], [&]() {
                val<2> newhyst = select(allocate[i], val<2>{0}, update_ctr(readh[i], ~badpred1[i]));
                ghyst[i].write(gindex[i], newhyst.fo1(), extra_cycle);
            });
        }

        val<1> line_end = block_entry >> (LINEINST - block_size);
        true_block = correct_pred | branch_dir[num_branch - 1] | line_end.fo1();
        true_block.fanout(hard<GHIST + NUMG * 2 + 2>{});
        execute_if(true_block, [&]() {
            next_pc.fanout(hard<2>{});
            global_history1 = (global_history1 << 1) ^ val<GHIST1>{next_pc >> 2};
            gfolds.update(val<PATHBITS>{next_pc >> 2});
        });

        num_branch = 0;
    }
};
