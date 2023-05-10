use std::{fs::File, io::BufReader};

use aleph_alpha_client::{
    cosine_similarity, Client, How, Modality, Prompt, Sampling, SemanticRepresentation, Stopping,
    Task, TaskCompletion, TaskSemanticEmbedding,
};
use dotenv::dotenv;
use image::ImageFormat;
use lazy_static::lazy_static;

lazy_static! {
    static ref AA_API_TOKEN: String = {
        // Use `.env` file if it exists
        let _ = dotenv();
        std::env::var("AA_API_TOKEN")
            .expect("AA_API_TOKEN environment variable must be specified to run tests.")
    };
}

#[tokio::test]
async fn completion_with_luminous_base() {
    // When
    let task = TaskCompletion::from_text("Hello", 1);

    let model = "luminous-base";
    let client = Client::new(&AA_API_TOKEN).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    eprintln!("{}", response.completion);

    // Then
    assert!(!response.completion.is_empty())
}

#[tokio::test]
async fn semanitc_search_with_luminous_base() {
    // Given
    let robot_fact = Prompt::from_text(
        "A robot is a machine—especially one programmable by a computer—capable of carrying out a \
        complex series of actions automatically.",
    );
    let pizza_fact = Prompt::from_text(
        "Pizza (Italian: [ˈpittsa], Neapolitan: [ˈpittsə]) is a dish of Italian origin consisting \
        of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, \
        and often various other ingredients (such as various types of sausage, anchovies, \
        mushrooms, onions, olives, vegetables, meat, ham, etc.), which is then baked at a high \
        temperature, traditionally in a wood-fired oven.",
    );
    let query = Prompt::from_text("What is Pizza?");
    let client = Client::new(&AA_API_TOKEN).unwrap();

    // When
    let robot_embedding_task = TaskSemanticEmbedding {
        prompt: robot_fact,
        representation: SemanticRepresentation::Document,
        compress_to_size: Some(128),
    };
    let robot_embedding = client
        .semantic_embedding(&robot_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;

    let pizza_embedding_task = TaskSemanticEmbedding {
        prompt: pizza_fact,
        representation: SemanticRepresentation::Document,
        compress_to_size: Some(128),
    };
    let pizza_embedding = client
        .semantic_embedding(&pizza_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;

    let query_embedding_task = TaskSemanticEmbedding {
        prompt: query,
        representation: SemanticRepresentation::Query,
        compress_to_size: Some(128),
    };
    let query_embedding = client
        .semantic_embedding(&query_embedding_task, &How::default())
        .await
        .unwrap()
        .embedding;
    let similarity_pizza = cosine_similarity(&query_embedding, &pizza_embedding);
    println!("similarity pizza: {similarity_pizza}");
    let similarity_robot = cosine_similarity(&query_embedding, &robot_embedding);
    println!("similarity robot: {similarity_robot}");

    // Then

    // The fact about pizza should be more relevant to the "What is Pizza?" question than a fact
    // about robots.
    assert!(similarity_pizza > similarity_robot);
}

#[tokio::test]
async fn complete_structured_prompt() {
    // Given
    let prompt = "Bot: Hello user!\nUser: Hello Bot, how are you doing?\nBot:";
    let stop_sequences = ["User:"];

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text(prompt),
        stopping: Stopping {
            maximum_tokens: 64,
            stop_sequences: &stop_sequences[..],
        },
        sampling: Sampling::MOST_LIKELY,
    };
    let model = "luminous-base";
    let client = Client::new(&AA_API_TOKEN).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert!(!response.completion.is_empty());
    assert!(!response.completion.contains("User:"));
}

#[tokio::test]
async fn describe_image_starting_from_a_path() {
    // Given
    let path_to_image = "tests/cat-chat-1641458.jpg";

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_vec(vec![
            Modality::from_image_path(path_to_image).unwrap(),
            Modality::from_text("A picture of "),
        ]),
        stopping: Stopping::from_maximum_tokens(10),
        sampling: Sampling::MOST_LIKELY,
    };
    let model = "luminous-base";
    let client = Client::new(&AA_API_TOKEN).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert!(response.completion.contains("cat"))
}

#[tokio::test]
async fn describe_image_starting_from_a_dyn_image() {
    // Given
    let path_to_image = "tests/cat-chat-1641458.jpg";
    let file = BufReader::new(File::open(path_to_image).unwrap());
    let format = ImageFormat::from_path(path_to_image).unwrap();
    let image = image::load(file, format).unwrap();

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_vec(vec![
            Modality::from_image(&image).unwrap(),
            Modality::from_text("A picture of "),
        ]),
        stopping: Stopping::from_maximum_tokens(10),
        sampling: Sampling::MOST_LIKELY,
    };
    let model = "luminous-base";
    let client = Client::new(&AA_API_TOKEN).unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();

    // Then
    eprintln!("{}", response.completion);
    assert!(response.completion.contains("cat"))
}
