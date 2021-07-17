use drl::plot::xy_plot::xy_scatter_plot;
use rand::Rng;
use random_choice::random_choice;

fn softmax(vals: &Vec<f64>, tau: f64) -> Vec<f64> {
    let e: f64 = 2.71828183;
    let powed_vals: Vec<f64> = vals.iter().map(|x| e.powf(x / tau)).collect();
    let sum: f64 = powed_vals.iter().sum();
    powed_vals.iter().map(|x| x / sum).collect()
}

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

#[derive(Debug, Default, Clone)]
struct Reward {
    frequence: i64,
    reward: f64,
}

fn main() {
    let mut rng = rand::thread_rng();

    let narms = 10;

    // hidden probabilities associated with each arm
    let probabilities: Vec<f64> = (0..narms).map(|_| rng.gen_range(0.0..1.0)).collect();
    println!("{:?}", probabilities);

    let arm_index: Vec<usize> = (0..narms).map(|x| x).collect();

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
        // choose an action according to softmax result
        let prob: Vec<f64> = record.iter().map(|x| x.reward).collect();
        let weights = softmax(&prob, 1.12);
        let choice = random_choice().random_choice_f64(&arm_index, &weights, 1);
        let choice = *choice[0];

        // get the reward of the choosen action
        let reward = get_reward(probabilities[choice], 10);

        // update experience
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
    xy_scatter_plot(
        String::from("n_arm_bandit_problem_with_softmax.svg"),
        rewards,
        -100.0,
        600.0,
        0.0,
        10.0,
        String::from("Plays"),
        String::from("Avg Reward"),
    );
}
