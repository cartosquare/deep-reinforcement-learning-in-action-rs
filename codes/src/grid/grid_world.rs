use crate::grid::grid_board::rand_pos;
use crate::grid::grid_board::GridBoard;
use std::collections::btree_map::BTreeMap;

#[derive(Debug)]
pub struct GridWorld {
    pub board: GridBoard,
}

#[derive(Debug)]
pub enum Action {
    UP,
    DOWN,
    LEFT,
    RIGHT,
}

impl GridWorld {
    pub fn new(size: i64, mode: String) -> GridWorld {
        let mut actual_size = size;
        if size < 4 {
            actual_size = 4;
        }

        let mut world = GridWorld {
            board: GridBoard::new(actual_size),
        };

        if mode == "static" {
            world.init_grid_static()
        } else if mode == "player" {
            world.init_grid_player();
        } else {
            world.init_grid_rand();
        }

        world
    }

    pub fn init_grid_static(&mut self) {
        // add pieces
        self.board
            .add_piece(String::from("Player"), String::from("P"), (0, 3));
        self.board
            .add_piece(String::from("Goal"), String::from("+"), (0, 0));
        self.board
            .add_piece(String::from("Pit"), String::from("-"), (0, 1));
        self.board
            .add_piece(String::from("Wall"), String::from("W"), (1, 1));
    }

    pub fn validate_move(&self, piece: String, move_direction: (i64, i64)) -> i64 {
        // 0 is valid, 1 invalid, 2 lost game
        let mut _outcome = 0;

        let pit = self.board.components.get("Pit").unwrap().pos;
        let wall = self.board.components.get("Wall").unwrap().pos;
        let target = self.board.components.get(&piece).unwrap().pos;
        let new_pos = (target.0 + move_direction.0, target.1 + move_direction.1);

        if new_pos == wall {
            1
        } else if new_pos == pit {
            2
        } else if new_pos.0 >= 0
            && new_pos.0 <= self.board.size - 1
            && new_pos.1 >= 0
            && new_pos.1 <= self.board.size - 1
        {
            0
        } else {
            1
        }
    }

    pub fn validate_board(&self) -> bool {
        let mut valid = true;

        // ensure no duplicate positions
        let mut all_position = BTreeMap::new();
        for c in self.board.components.iter() {
            *all_position.entry(c.1.pos).or_insert(0) += 1;
        }
        if all_position.len() < 4 {
            valid = false;
        }

        // ensure not dead position
        let player = self.board.components.get("Player").unwrap().pos;
        let goal = self.board.components.get("Goal").unwrap().pos;
        let corner_positions: Vec<(i64, i64)> = vec![
            (0, 0),
            (0, self.board.size - 1),
            (self.board.size - 1, 0),
            (self.board.size - 1, self.board.size - 1),
        ];
        let move_directions: Vec<(i64, i64)> = vec![(0, 1), (-1, 0), (1, 0), (-1, 0)];

        if corner_positions.iter().any(|&x| x == player) {
            let move_status: Vec<i64> = move_directions
                .iter()
                .map(|&x| self.validate_move(String::from("Player"), x))
                .collect();
            if !move_status.iter().any(|&x| x == 0) {
                valid = false
            }
        }

        if corner_positions.iter().any(|&x| x == goal) {
            let move_status: Vec<i64> = move_directions
                .iter()
                .map(|&x| self.validate_move(String::from("Goal"), x))
                .collect();
            if !move_status.iter().any(|&x| x == 0) {
                valid = false
            }
        }

        valid
    }

    pub fn init_grid_player(&mut self) {
        self.init_grid_static();
        let mut player = self.board.components.get_mut("Player").unwrap();
        (*player).pos = rand_pos(0, self.board.size);

        if !self.validate_board() {
            self.init_grid_player();
        }
    }

    pub fn init_grid_rand(&mut self) {
        self.board.add_piece(
            String::from("Player"),
            String::from("P"),
            rand_pos(0, self.board.size),
        );
        self.board.add_piece(
            String::from("Goal"),
            String::from("+"),
            rand_pos(0, self.board.size),
        );
        self.board.add_piece(
            String::from("Pit"),
            String::from("-"),
            rand_pos(0, self.board.size),
        );
        self.board.add_piece(
            String::from("Wall"),
            String::from("W"),
            rand_pos(0, self.board.size),
        );

        if !self.validate_board() {
            self.init_grid_rand();
        }
    }

    pub fn display(&self) {
        self.board.render();
    }

    pub fn make_move(&mut self, action: Action) {
        let move_direction: (i64, i64);
        match action {
            Action::UP => move_direction = (-1, 0),
            Action::DOWN => move_direction = (1, 0),
            Action::LEFT => move_direction = (0, -1),
            Action::RIGHT => move_direction = (0, 1),
        }

        let move_status = self.validate_move(String::from("Player"), move_direction);
        if move_status == 0 || move_status == 2 {
            let mut player = self.board.components.get_mut("Player").unwrap();
            (*player).pos = (
                (*player).pos.0 + move_direction.0,
                (*player).pos.1 + move_direction.1,
            );
        }
    }

    pub fn reward(&self) -> f64 {
        let pit_pos = self.board.components.get("Pit").unwrap().pos;
        let goal_pos = self.board.components.get("Goal").unwrap().pos;
        let player_pos = self.board.components.get("Player").unwrap().pos;
        if pit_pos == player_pos {
            -10.0
        } else if goal_pos == player_pos {
            10.0
        } else {
            -1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::grid::grid_world::*;

    #[test]
    fn test_create_grid_world_static() {
        let world = GridWorld::new(5, String::from("static"));
        world.board.render();
        assert_eq!(world.validate_board(), true);
    }

    #[test]
    fn test_crate_grid_world_player() {
        let world = GridWorld::new(5, String::from("player"));
        world.board.render();
        assert_eq!(world.validate_board(), true);
    }

    #[test]
    fn test_crate_grid_world_random() {
        let world = GridWorld::new(5, String::from("random"));
        world.board.render();
        assert_eq!(world.validate_board(), true);
    }

    #[test]
    fn test_move() {
        let mut world = GridWorld::new(4, String::from("static"));
        world.display();

        world.make_move(Action::DOWN);
        world.make_move(Action::DOWN);
        world.make_move(Action::LEFT);
        world.make_move(Action::LEFT);
        world.make_move(Action::LEFT);
        world.make_move(Action::UP);
        world.make_move(Action::UP);
        world.make_move(Action::RIGHT);
        world.display();

        println!("{}", world.reward());
        println!("{:?}", world.board.render_array());
    }

    #[test]
    fn test_crate_validate_board0() {
        let mut world = GridWorld::new(5, String::from("static"));
        let mut player = world.board.components.get_mut("Player").unwrap();
        (*player).pos = (0, 0);

        let mut wall = world.board.components.get_mut("Wall").unwrap();
        (*wall).pos = (0, 1);

        let mut pit = world.board.components.get_mut("Pit").unwrap();
        (*pit).pos = (1, 0);

        let mut goal = world.board.components.get_mut("Goal").unwrap();
        (*goal).pos = (4, 4);

        world.board.render();
        assert_eq!(world.validate_board(), false);

        let mut goal = world.board.components.get_mut("Goal").unwrap();
        (*goal).pos = (0, 0);
        let mut player = world.board.components.get_mut("Player").unwrap();
        (*player).pos = (4, 4);

        world.board.render();
        assert_eq!(world.validate_board(), false);
    }

    #[test]
    fn test_crate_validate_board1() {
        let mut world = GridWorld::new(5, String::from("static"));
        let mut player = world.board.components.get_mut("Player").unwrap();
        (*player).pos = (0, 4);

        let mut wall = world.board.components.get_mut("Wall").unwrap();
        (*wall).pos = (0, 3);

        let mut pit = world.board.components.get_mut("Pit").unwrap();
        (*pit).pos = (1, 4);

        world.board.render();
        assert_eq!(world.validate_board(), false);

        let mut goal = world.board.components.get_mut("Goal").unwrap();
        (*goal).pos = (0, 4);
        let mut player = world.board.components.get_mut("Player").unwrap();
        (*player).pos = (0, 0);

        world.board.render();
        assert_eq!(world.validate_board(), false);
    }

    #[test]
    fn test_crate_validate_board2() {
        let mut world = GridWorld::new(5, String::from("static"));
        let mut player = world.board.components.get_mut("Player").unwrap();
        (*player).pos = (4, 0);

        let mut wall = world.board.components.get_mut("Wall").unwrap();
        (*wall).pos = (4, 1);

        let mut pit = world.board.components.get_mut("Pit").unwrap();
        (*pit).pos = (3, 0);

        world.board.render();
        assert_eq!(world.validate_board(), false);

        let mut goal = world.board.components.get_mut("Goal").unwrap();
        (*goal).pos = (4, 0);

        let mut player = world.board.components.get_mut("Player").unwrap();
        (*player).pos = (0, 0);

        world.board.render();
        assert_eq!(world.validate_board(), false);
    }

    #[test]
    fn test_crate_validate_board3() {
        let mut world = GridWorld::new(5, String::from("static"));
        let mut player = world.board.components.get_mut("Player").unwrap();
        (*player).pos = (4, 4);

        let mut wall = world.board.components.get_mut("Wall").unwrap();
        (*wall).pos = (3, 4);

        let mut pit = world.board.components.get_mut("Pit").unwrap();
        (*pit).pos = (4, 3);

        world.board.render();
        assert_eq!(world.validate_board(), false);

        let mut goal = world.board.components.get_mut("Goal").unwrap();
        (*goal).pos = (4, 4);

        let mut player = world.board.components.get_mut("Player").unwrap();
        (*player).pos = (0, 0);

        world.board.render();
        assert_eq!(world.validate_board(), false);
    }
}
