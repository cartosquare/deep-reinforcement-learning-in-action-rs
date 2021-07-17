use rand::Rng;
use std::collections::HashMap;

// base unit of board
#[derive(Debug)]
pub struct BoardPiece {
    pub name: String,
    pub code: String,
    pub pos: (i64, i64),
}

pub fn rand_pos(low: i64, high: i64) -> (i64, i64) {
    let mut rng = rand::thread_rng();
    (rng.gen_range(low..high), rng.gen_range(low..high))
}

#[derive(Debug)]
pub struct GridBoard {
    pub size: i64,
    pub components: HashMap<String, BoardPiece>,
}

impl GridBoard {
    pub fn new(size: i64) -> GridBoard {
        GridBoard {
            size,
            components: HashMap::new()
        }
    }

    pub fn add_piece(&mut self, name: String, code: String, pos: (i64, i64)) {
        let piece = BoardPiece{
            name: name.clone(), 
            code, pos
        };
        self.components.insert(name.clone(), piece);
    }

    pub fn render(&self) {
        for i in 0..self.size {
            for j in 0..self.size {
                let mut special_pos = false;
                for (_name, piece) in self.components.iter() {
                    if (i, j) == piece.pos {
                        print!(" {} ", piece.code);
                        special_pos = true;
                        break;
                    }
                }
                if !special_pos {
                    print!(" * ");
                }
            }
            println!("");
        }
        println!("");
    }

    pub fn render_array(&self) -> Vec<f64> {
        let len = self.size as usize;
        let mut pattern: Vec<f64> =  vec![0.0; len * len * self.components.len()];
        let mut frame_index: usize = 0;
        let frame_size = len * len;
        for (_name, piece) in self.components.iter() {
            pattern[frame_index * frame_size + piece.pos.0 as usize * len + piece.pos.1 as usize] = 1.0;
            frame_index += 1;
        }
        pattern
    }
}

#[cfg(test)]
mod tests {
    use crate::grid::grid_board::*;

    #[test]
    fn test_board_render() {
        let mut board = GridBoard::new(5);
        board.add_piece(String::from("Player"), String::from("P"), (3, 3));
        board.add_piece(String::from("Goal"), String::from("O"), (1, 1));
        board.render();
    }
}