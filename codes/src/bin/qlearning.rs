use drl::grid::grid_world::{Action, GridWorld};
use drl::plot::xy_plot::xy_scatter_plot;
use rand::Rng;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

fn net(vs: &nn::Path, input: i64, hidden1: i64, hidden2: i64, output: i64) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            input,
            hidden1,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer2",
            hidden1,
            hidden2,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer3",
            hidden2,
            output,
            Default::default(),
        ))
}

fn train_model(vs: &nn::VarStore, model: &impl Module, epochs: i64, mode: String) {
    let mut rng = rand::thread_rng();

    let mut epsilon = 1.0;
    let gamma = 0.9;

    // average loss of each epoch
    let mut losses: Vec<(f64, f64)> = vec![];

    // optimizer
    let mut optimizer = nn::Adam::default().build(&vs, 0.001).unwrap();
    // the main loop
    for i in 0..epochs {
        // for each epoch, we start a new game
        let mut game = GridWorld::new(4, mode.clone());

        // after we create the game, we extract the state information
        // and add a small amount of noise
        let state = game.board.render_array();
        let mut state = Tensor::of_slice(&state).to_kind(Kind::Float)
            + Tensor::rand(&vec![1, 64], (Kind::Float, Device::Cpu)) / 10.0;

        // steps in current epoch
        let mut status = 1;
        let mut avg_loss = 0.0;
        let mut steps = 0;
        while status == 1 {
            let q_value = model.forward(&state);

            // selects an action using the epsilon-greedy method
            let action_index: i64;
            if rng.gen_range(0.0..1.0) > epsilon {
                action_index = i64::from(q_value.argmax(1, true));
            } else {
                action_index = rng.gen_range(0..4);
            }

            let action = match action_index {
                0 => Action::UP,
                1 => Action::DOWN,
                2 => Action::LEFT,
                3 => Action::RIGHT,
                _ => Action::UP,
            };

            // take the action
            game.make_move(action);

            // .. and we get a new state
            let state2 = game.board.render_array();
            let state2 = Tensor::of_slice(&state2).to_kind(Kind::Float)
                + Tensor::rand(&vec![1, 64], (Kind::Float, Device::Cpu)) / 10.0;

            // .. and get the corresponding reward
            let reward = game.reward();

            let new_q_value = tch::no_grad(|| model.forward(&state2));
            let max_q_value = new_q_value.max();

            // q-learning schema:
            // Q(S_t, A_t) = Q(S_t, A_t) + alpha * (R_t+1 + discount * maxQ(S_t+1, a) - Q(S_t, A_t))
            // Here the nn model learns the reward
            let mut y = reward;
            if reward == -1.0 {
                // game is not stopped, afterwards rewards count
                y += gamma * f64::from(max_q_value);
            }

            // fix the pred reward
            let mut q_value_fixed = Vec::<f64>::from(&q_value.copy());
            q_value_fixed[action_index as usize] = y;
            let y_target = Tensor::of_slice(&q_value_fixed).to_kind(Kind::Float);

            let loss = q_value.mse_loss(&y_target, tch::Reduction::Sum);
            optimizer.zero_grad();
            optimizer.backward_step(&loss);

            // update state
            state = state2;
            if reward != -1.0 {
                // game finish
                status = 0;
            }

            //
            avg_loss = avg_loss + f64::from(loss);
            steps = steps + 1;
        }

        losses.push((i as f64, avg_loss / steps as f64));
        if epsilon > 0.1 {
            epsilon -= 1.0 / epochs as f64;
        }
    }
    xy_scatter_plot(
        String::from("listing.svg"),
        losses,
        (epochs - 1000) as f64,
        (epochs + 1000) as f64,
        0.0,
        10.0,
        String::from("epoch"),
        String::from("loss"),
    );
}

fn test_model(model: &impl Module, display: bool, mode: String) -> bool {
    let mut game = GridWorld::new(4, mode.clone());
    let state = game.board.render_array();
    let mut state = Tensor::of_slice(&state).to_kind(Kind::Float)
        + Tensor::rand(&vec![1, 64], (Kind::Float, Device::Cpu)) / 10.0;
    if display {
        println!("Initial State: ");
        game.display();
    }

    let mut status = 1;
    let mut i = 0;
    while status == 1 {
        let qeval = model.forward(&state);
        let action = qeval.argmax(1, true);
        let action = match i64::from(action) {
            0 => Action::UP,
            1 => Action::DOWN,
            2 => Action::LEFT,
            3 => Action::RIGHT,
            _ => Action::UP,
        };
        if display {
            println!("Move #: {}; Taking action: {:?}", i, action);
        }
        game.make_move(action);
        let state2 = game.board.render_array();
        state = Tensor::of_slice(&state2).to_kind(Kind::Float)
            + Tensor::rand(&vec![1, 64], (Kind::Float, Device::Cpu)) / 10.0;
        if display {
            game.display();
        }

        let reward = game.reward();
        if reward != -1.0 {
            if reward > 0.0 {
                status = 2;
                if display {
                    println!("Game won! Reward: {}", reward);
                } else {
                    status = 0;
                    if display {
                        println!("Game Lost. Reward: {}", reward);
                    }
                }
            }
        }

        i = i + 1;
        if i > 15 {
            if display {
                println!("Game lost; too many moves.");
            }
            break;
        }
    }
    status == 2
}

fn main() {
    // nn
    let vs = nn::VarStore::new(Device::Cpu);
    let model = net(&vs.root(), 64, 150, 100, 4);

    train_model(&vs, &model, 1000, String::from("static"));

    test_model(&model, true, String::from("static"));
}
