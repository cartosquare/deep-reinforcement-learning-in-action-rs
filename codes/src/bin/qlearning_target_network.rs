use drl::grid::grid_world::{Action, GridWorld};
use drl::plot::xy_plot::xy_scatter_plot;
use rand::distributions::Slice;
use rand::{thread_rng, Rng};
use std::collections::VecDeque;
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
    let mut rng = thread_rng();

    let mut epsilon = 1.0;
    let gamma = 0.9;

    // average loss of each epoch
    let mut losses: Vec<(f64, f64)> = vec![];
    let mut loss_min = 1000000.0;
    let mut loss_max = 0.0;

    let memory_size = 1000;
    let batch_size = 200;

    // state-action-reward-state-done
    let mut replay: VecDeque<(Tensor, i64, f64, Tensor, i64)> = VecDeque::with_capacity(memory_size);
    let max_move = 50;

    // update frequency for synchronizing the
    // target model parameters
    let sync_frequence = 500;
    let mut target_model = model.clone();
    let mut j = 0;

    // optimizer
    let mut optimizer = nn::Adam::default().build(&vs, 0.001).unwrap();
    // the main loop
    let mut steps = 0;
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
        let mut moves = 0;
        while status == 1 {
            j = j + 1;
            moves = moves + 1;

            let q_value = tch::no_grad(|| model.forward(&state));

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
            game.make_move(action.clone());

            // .. and we get a new state
            let state2 = game.board.render_array();
            let state2 = Tensor::of_slice(&state2).to_kind(Kind::Float)
                + Tensor::rand(&vec![1, 64], (Kind::Float, Device::Cpu)) / 10.0;

            // .. and get the corresponding reward
            let reward = game.reward();

            let done = reward > 0.0;

            // adds the experience to the experience replay list
            replay.push_back((
                state.copy(),
                action_index,
                reward,
                state2.copy(),
                done as i64,
            ));

            state = state2;

            // if the replay list is at least as long
            // as the mini-batch size,
            // begins the mini-batch training
            if replay.len() > batch_size {
                let replay_slice = replay.iter().collect::<Vec<_>>();
                // println!("replay slice: {}, {:?}", replay_slice.len(), replay_slice[0]);

                let replay_slice = Slice::new(&replay_slice).unwrap();
                let minibatch = (&mut rng)
                    .sample_iter(&replay_slice)
                    .take(batch_size)
                    .collect::<Vec<_>>();
                // println!("minibatch: {}, {:?}", minibatch.len(), minibatch[0]);

                let state1_batch =
                    Tensor::cat(&minibatch.iter().map(|&x| x.0.copy()).collect::<Vec<_>>(), 0);
                // println!("state batch: {:?}",state1_batch.size());

                let action_batch = minibatch.iter().map(|&x| x.1).collect::<Vec<_>>();
                let action_batch = Tensor::of_slice(&action_batch).to_kind(Kind::Int64);

                let reward_batch = minibatch.iter().map(|&x| x.2).collect::<Vec<_>>();
                let reward_batch = Tensor::of_slice(&reward_batch).to_kind(Kind::Float);

                let state2_batch =
                    Tensor::cat(&minibatch.iter().map(|&x| x.3.copy()).collect::<Vec<_>>(), 0);

                let done_batch = minibatch.iter().map(|&x| x.4 as f64).collect::<Vec<_>>();
                let done_batch = Tensor::of_slice(&done_batch).to_kind(Kind::Float);

                // recomputes q values for the mini-batch of states to get gradients
                // q1 size: [batch_size, 4]
                let q1 = model.forward(&state1_batch);
                // println!("q1 size: {:?}", q1.size());

                // compute q values for the next state, but not get gradients
                // q2 size: [batch_size, 4]
                let q2 = tch::no_grad(|| target_model.forward(&state2_batch));
                // println!("q2 size: {:?}", q2.size());

                // println!("action size: {:?}", action_batch.size());

                // action_batch size: [batch_size]
                // q1_selected size: [batch_size]
                let q1_selected = q1.gather(1, &action_batch.unsqueeze(1), false).squeeze();

                // println!("q1 selected size: {:?}", q1_selected.size());

                // y_target size: [batch_size]
                let y_target: Tensor = reward_batch + gamma * ((1 - done_batch) * q2.max_dim(1, false).0);
                // println!("target size: {:?}", y_target.size());

                let loss = q1_selected.mse_loss(&y_target.detach(), tch::Reduction::Sum);
                // println!("loss size: {:?}", loss);
                optimizer.zero_grad();
                optimizer.backward_step(&loss);

                //
                let l = f64::from(loss);
                if l < loss_min {
                    loss_min = l;
                }
                if l > loss_max {
                    loss_max = l;
                }
                losses.push((steps as f64, l));
                steps = steps + 1;

                if j % sync_frequence == 0 {
                    target_model = model.clone();
                }
            }

            if reward != -1.0 || moves > max_move {
                // game finish
                status = 0;
            }
        }

        if epsilon > 0.1 {
            epsilon -= 1.0 / epochs as f64;
        }

        if (i + 1) % 100 == 0 {
            println!("#epoch {}, epsilon: {}", i + 1, epsilon);
        }
    }

    xy_scatter_plot(
        String::from("qlearning_target_model.svg"),
        losses,
        -100.0,
        (steps + 1000) as f64,
        loss_min,
        loss_max,
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
                }
            } else {
                status = 0;
                if display {
                    println!("Game Lost. Reward: {}", reward);
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

    let max_games = 1000;
    let mut wins = 0;

    for _i in 0..max_games {
        if test_model(&model, true, String::from("static")) {
            wins = wins + 1;
        }
    }
    
    let win_rate = wins as f64 / max_games as f64;
    println!("Games played: {}, # of wins: {}", max_games, wins);
    println!("Win percentage: {}", win_rate);
}
