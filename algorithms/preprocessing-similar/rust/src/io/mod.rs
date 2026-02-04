// ------------------------------------------------------------
// IMPORTS
// ------------------------------------------------------------
// Serde is a Rust library for converting data
// between formats (like CSV, JSON, etc.) and Rust structs.
//
// Deserialize: "convert from CSV text into a Rust struct."
use serde::Deserialize;

// Standard Rust trait used for returning errors.
use std::error::Error;

// ------------------------------------------------------------
// STRUCT: Passenger
// ------------------------------------------------------------
//
//
// Each row in the Titanic dataset represents ONE passenger.
//
// This struct defines what one passenger looks like.
//

#[derive(Debug, Deserialize, Clone)]
pub struct Passenger {
    // --------------------------------------------------------
    // PassengerId column
    // --------------------------------------------------------
    //
    // This is a unique ID number for each passenger.
    //
    // u32 means: an unsigned 32-bit integer (only positive).
    //
    #[serde(rename = "PassengerId")]
    pub passenger_id: u32,

    // --------------------------------------------------------
    // Survived column
    // --------------------------------------------------------
    //
    // This is the label we often want to predict:
    //
    // 0 = did NOT survive
    // 1 = survived
    //
    // We store it as u8 (small integer).
    //
    #[serde(rename = "Survived")]
    pub survived: u8,

    // --------------------------------------------------------
    // Pclass column
    // --------------------------------------------------------
    //
    // Passenger class (social/economic class):
    //
    // 1 = First class
    // 2 = Second class
    // 3 = Third class
    //
    #[serde(rename = "Pclass")]
    pub pclass: u8,

    // --------------------------------------------------------
    // Name column
    // --------------------------------------------------------
    //
    // Passenger's full name.
    //
    // String means heap-allocated UTF-8 text.
    //
    #[serde(rename = "Name")]
    pub name: String,

    // --------------------------------------------------------
    // Sex column
    // --------------------------------------------------------
    //
    // Passenger gender stored as text:
    //
    // "male" or "female"
    //
    // Later, for machine learning, we may convert this
    // into numeric form.
    //
    #[serde(rename = "Sex")]
    pub sex: String,

    // --------------------------------------------------------
    // Age column
    // --------------------------------------------------------
    //
    // Age is NOT always present in the dataset.
    //
    // Rust does not allow null values.
    //
    // So we use Option<f32>:
    //
    // Some(age) → age exists
    // None      → missing value
    //
    #[serde(rename = "Age")]
    pub age: Option<f32>,

    // --------------------------------------------------------
    // SibSp column
    // --------------------------------------------------------
    //
    // Number of siblings/spouses aboard.
    //
    #[serde(rename = "SibSp")]
    pub sibsp: u8,

    // --------------------------------------------------------
    // Parch column
    // --------------------------------------------------------
    //
    // Number of parents/children aboard.
    //
    #[serde(rename = "Parch")]
    pub parch: u8,

    // --------------------------------------------------------
    // Ticket column
    // --------------------------------------------------------
    //
    // Ticket identifier (often alphanumeric).
    //
    #[serde(rename = "Ticket")]
    pub ticket: String,

    // --------------------------------------------------------
    // Fare column
    // --------------------------------------------------------
    //
    // The amount paid for the ticket.
    //
    // We use f64 because money values may include decimals
    // and need higher precision.
    //
    #[serde(rename = "Fare")]
    pub fare: f64,

    // --------------------------------------------------------
    // Cabin column
    // --------------------------------------------------------
    //
    // Cabin number is missing for MANY passengers.
    //
    // So we use Option<String>.
    //
    #[serde(rename = "Cabin")]
    pub cabin: Option<String>,

    // --------------------------------------------------------
    // Embarked column
    // --------------------------------------------------------
    //
    // Port where the passenger boarded:
    //
    // C = Cherbourg
    // Q = Queenstown
    // S = Southampton
    //
    // Some passengers are missing this value.
    //
    #[serde(rename = "Embarked")]
    pub embarked: Option<String>,
}

// ------------------------------------------------------------
// FUNCTION: load_titanic
// ------------------------------------------------------------
//
// This function loads the Titanic CSV file into memory.
//
// INPUT:
//   path "data/titanic.csv"
//
// OUTPUT:
//   Result<Vec<Passenger>, Box<dyn Error>>
//      Result is an enum built into Rust
//      Result is an enum built into Rust
//
// Meaning:
//
//   - If successful → Ok(vector_of_passengers)
//   - If something fails → Err(error_message)
//
// Rust forces you to handle errors safely instead of crashing.
//
pub fn load_titanic(path: &str) -> Result<Vec<Passenger>, Box<dyn Error>> {
    // --------------------------------------------------------
    // Step 1: Create a CSV reader
    // --------------------------------------------------------
    //
    // csv::ReaderBuilder allows configuration of the parser.
    //
    // has_headers(true):
    //   tells Rust that the first row contains column names.
    //
    // flexible(true):
    //   allows slightly inconsistent row lengths (more tolerant).
    //
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)?; // ? means "return the error if file fails"

    // --------------------------------------------------------
    // Step 2: Create an empty vector to store passengers
    // --------------------------------------------------------
    //
    // Vec<T> is Rust's growable list type.
    //
    let mut passengers: Vec<Passenger> = Vec::new();

    // --------------------------------------------------------
    // Step 3: Read each row of the CSV file
    // --------------------------------------------------------
    //
    // reader.deserialize() automatically converts each row
    // into a Passenger struct (because we derived Deserialize).
    //
    for row_result in reader.deserialize() {
        // row_result is a Result<Passenger, Error>

        // If the row is valid → extract Passenger
        // If the row is invalid → return the error
        let passenger: Passenger = row_result?;

        // Add this passenger to our vector
        passengers.push(passenger);
    }

    // --------------------------------------------------------
    // Step 4: Return the completed passenger list
    // --------------------------------------------------------
    //
    // Ok(...) means the function succeeded.
    //
    Ok(passengers)
}
