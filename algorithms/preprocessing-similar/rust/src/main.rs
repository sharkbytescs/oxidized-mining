mod io;

fn main() {
    println!("Hello, world!");

    let path = "data/titanic.csv";

    let passengers = io::load_titanic(path).expect("Failed to load Titanic dataset");

    println!("Loaded {} passengers!", passengers.len());
}
