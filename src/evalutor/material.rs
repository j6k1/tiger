use usiagent::rule::{Rule, State};
use usiagent::shogi::{MochigomaCollections, Teban};

const PIECE_SCORE_MAP:[i32; 29] = [
    90 * 9 / 10,
    315 * 9 / 10,
    405 * 9 / 10,
    495 * 9 / 10,
    540 * 9 / 10,
    855 * 9 / 10,
    990 * 9 / 10,
    15000 * 9 / 10,
    540 * 9 / 10,
    540 * 9 / 10,
    540 * 9 / 10,
    540 * 9 / 10,
    945 * 9 / 10,
    1395 * 9 / 10,
    -90 * 9 / 10,
    -315 * 9 / 10,
    -405 * 9 / 10,
    -495 * 9 / 10,
    -540 * 9 / 10,
    -855 * 9 / 10,
    -990 * 9 / 10,
    -15000 * 9 / 10,
    -540 * 9 / 10,
    -540 * 9 / 10,
    -540 * 9 / 10,
    -540 * 9 / 10,
    -945 * 9 / 10,
    -1395 * 9 / 10,
    0
];

const HAND_SCORE_MAP: [i32; 7] = [
    90,315,405,405,540,855,990
];

pub struct Evalutor {
}
impl Evalutor {
    pub fn new() -> Self {
        Evalutor {}
    }
    pub fn evalute(&self,teban:Teban,state:&State,mc:&MochigomaCollections) -> i32 {
        let banmen = state.get_banmen();

        let mut score = 0;

        for y in 0..9 {
            for x in 0..9 {
                let (x,y,s) = if teban == Teban::Sente {
                    (x,y,1)
                } else {
                    (8-x,8-y,-1)
                };

                score += s * PIECE_SCORE_MAP[banmen.0[y][x] as usize];
            }
        }

        match mc {
            &MochigomaCollections::Pair(ref ms,ref mg) if teban == Teban::Sente => {
                for (m,c) in ms.iter() {
                    score += HAND_SCORE_MAP[m as usize] * c as i32;
                }
                for (m,c) in mg.iter() {
                    score -= HAND_SCORE_MAP[m as usize] * c as i32;
                }
            },
            &MochigomaCollections::Pair(ref ms, ref mg) => {
                for (m,c) in mg.iter() {
                    score += HAND_SCORE_MAP[m as usize] * c as i32;
                }
                for (m,c) in ms.iter() {
                    score -= HAND_SCORE_MAP[m as usize] * c as i32;
                }
            },
            _ => ()
        }

        match teban {
            Teban::Sente => {
                score += state.get_part().sente_self_board.iter().map(|p| {
                    Rule::control_count(teban,state,p) as i32
                }).fold(0,|acc,c| acc + c);

                score -= state.get_part().gote_self_board.iter().map(|p| {
                    Rule::control_count(teban,state,p) as i32
                }).fold(0,|acc,c| acc + c);
            },
            Teban::Gote => {
                score += state.get_part().gote_self_board.iter().map(|p| {
                    Rule::control_count(teban,state,p) as i32
                }).fold(0,|acc,c| acc + c);

                score -= state.get_part().sente_self_board.iter().map(|p| {
                    Rule::control_count(teban,state,p) as i32
                }).fold(0,|acc,c| acc + c);
            }
        }

        score
    }
}