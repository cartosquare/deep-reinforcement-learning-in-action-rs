use drl::plot::xy_plot::xy_scatter_plot;
use rand::Rng;
use std::collections::HashMap;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

// Environment
struct ContextBandit {
    arms: usize,
    // state-action -> reward
    // key = state * 10 + action(since we only get 10 states and 10 actions)
    bandit_matrix: HashMap<usize, f64>,
    state: usize,
}

impl ContextBandit {
    pub fn new(arms: usize) -> ContextBandit {
        ContextBandit {
            arms,
            state: ContextBandit::sample_state(arms),
            bandit_matrix: ContextBandit::init_distribution(arms, arms),
        }
    }

    fn print(&self) {
        println!(
            "arms: {}, state: {}, matrix: {:?}",
            self.arms, self.state, self.bandit_matrix
        );
    }

    fn key(state: usize, action: usize) -> usize {
        state * 10 + action
    }

    fn init_distribution(states: usize, actions: usize) -> HashMap<usize, f64> {
        let mut rng = rand::thread_rng();
        let mut matrix: HashMap<usize, f64> = HashMap::new();
        for s in 0..states {
            for a in 0..actions {
                let prob = rng.gen_range(0.0..1.0);
                matrix.insert(ContextBandit::key(s, a), prob);
            }
        }
        matrix
    }

    fn sample_state(states: usize) -> usize {
        let mut rng = rand::thread_rng();
        rng.gen_range(0..states)
    }

    fn get_arms(&self) -> usize {
        self.arms
    }

    fn get_state(&self) -> usize {
        self.state
    }

    fn update_state(&mut self) {
        self.state = ContextBandit::sample_state(self.arms);
    }

    // simulate get reward
    fn reward(&self, proability: f64) -> f64 {
        let mut rng = rand::thread_rng();

        let mut reward = 0.0;
        for _ in 0..self.arms {
            if rng.gen_range(0.0..1.0) < proability {
                reward += 1.0;
            }
        }
        reward
    }

    fn get_reward(&self, arm: usize) -> f64 {
        let key = ContextBandit::key(self.get_state(), arm);
        self.reward(*self.bandit_matrix.get(&key).unwrap())
    }

    fn choose_arm(&mut self, arm: usize) -> f64 {
        // choose an arm
        // returns a reward
        // and update the state
        let reward = self.get_reward(arm);
        self.update_state();
        reward
    }
}

fn one_hot(n: usize, pos: usize) -> Tensor {
    let v: Vec<f64> = (0..n).map(|x| if pos == x { 1.0 } else { 0.0 }).collect();
    Tensor::of_slice(&v).to_kind(Kind::Float)
}

fn net(vs: &nn::Path, arms: i64, hidden: i64) -> impl Module {
    nn::seq()
        .add(nn::linear(vs / "layer1", arms, hidden, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer2", hidden, arms, Default::default()))
        .add_fn(|xs| xs.relu())
}

fn train(env: &mut ContextBandit, epochs: i64, learning_rate: f64) -> Vec<(f64, f64)> {
    let vs = nn::VarStore::new(Device::Cpu);
    let net = net(&vs.root(), env.get_arms() as i64, 100);
    let mut optimizer = nn::Adam::default().build(&vs, learning_rate).unwrap();

    let mut rewards: Vec<(f64, f64)> = vec![];
    for i in 0..epochs {
        let current_state = one_hot(env.get_arms(), env.get_state());
        let y_pred = net.forward(&current_state);
        // softmax and choose new action probabilistically
        let choice = y_pred.softmax(0, Kind::Float).multinomial(1, true);
        let choice = i64::from(choice) as usize;
        let current_reward = env.choose_arm(choice);

        // fix the pred reward
        let mut pred = Vec::<f64>::from(&y_pred.copy());
        pred[choice] = current_reward;
        let y_target = Tensor::of_slice(&pred).to_kind(Kind::Float);

        if i == 0 {
            rewards.push((i as f64, current_reward));
        } else {
            let mean_reward =
                (i as f64 * rewards.last().unwrap().1 + current_reward) / (i + 1) as f64;
            rewards.push((i as f64, mean_reward));
        }

        let loss = y_pred.mse_loss(&y_target, tch::Reduction::Sum);
        optimizer.zero_grad();
        optimizer.backward_step(&loss);
    }
    rewards
}

fn main() {
    let mut env = ContextBandit::new(10);
    // env.print();
    // let state = env.get_state();
    // println!("state: {}", state);
    // let reward = env.choose_arm(1);
    // println!("reward: {}", reward);

    // let t = one_hot(10, 5);
    // println!("one hot: {:?}", t);
    // t.print();

    let rewards = train(&mut env, 5000, 0.01);
    xy_scatter_plot(
        String::from("contextual_n_arm_bandit_problem.svg"),
        rewards,
        -1000.0,
        600.0,
        0.0,
        10.0,
        String::from("Plays"),
        String::from("Avg Reward"),
    );
}
