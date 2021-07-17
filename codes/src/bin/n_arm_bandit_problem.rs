use drl::plot::xy_plot::xy_scatter_plot;
use rand::Rng;

// simulate get reward
fn get_reward(proability: f64, n: i64) -> f64 {
    let mut rng = rand::thread_rng();

    let mut reward = 0.0;
    for _ in 0..n {
        if rng.gen_range(0.0..1.0) < proability {
            reward += 1.0;
        }
    }
    reward
}

// calcuate avg reward
fn update_record(record: &mut Vec<Reward>, action: usize, reward: f64) {
    let new_reward = (record[action].frequence as f64 * record[action].reward + reward)
        / (record[action].frequence + 1) as f64;
    record[action].frequence += 1;
    record[action].reward = new_reward;
}

// get best action index
fn get_best_arm(record: &Vec<Reward>) -> usize {
    let mut max_index = 0;
    let mut max_value = 0.0;
    for (index, value) in record.iter().enumerate() {
        if max_value < value.reward {
            max_value = value.reward;
            max_index = index;
        }
    }
    max_index
}

#[derive(Debug, Default, Clone)]
struct Reward {
    frequence: i64,
    reward: f64,
}

fn main() {
    let mut rng = rand::thread_rng();

    // 摇杆个数
    let narms = 10;

    // epsilon-greedy系数
    let epsilon = 0.2;

    // hidden probabilities associated with each arm
    let probabilities: Vec<f64> = (0..narms).map(|_| rng.gen_range(0.0..1.0)).collect();
    println!("{:?}", probabilities);

    // state-action-reward
    let mut record: Vec<Reward> = vec![
        Reward {
            frequence: 0,
            reward: 0.0,
        };
        10
    ];

    // accumulate rewards
    let mut rewards: Vec<(f64, f64)> = vec![];
    for i in 0..500 {
        let choice: usize;
        if rng.gen_range(0.0..1.0) > epsilon {
            choice = get_best_arm(&record);
        } else {
            choice = rng.gen_range(0..narms);
        }

        let reward = get_reward(probabilities[choice], 10);
        update_record(&mut record, choice, reward);
        if i == 0 {
            rewards.push((i as f64, reward));
        } else {
            let mean_reward = (i as f64 * rewards.last().unwrap().1 + reward) / (i + 1) as f64;
            rewards.push((i as f64, mean_reward));
        }
    }
    println!("{:?}", rewards);

    // plot graph
    // plot graph
    xy_scatter_plot(
        String::from("n_arm_bandit_problem.svg"),
        rewards,
        -100.0,
        600.0,
        0.0,
        10.0,
        String::from("Plays"),
        String::from("Avg Reward"),
    )
}
