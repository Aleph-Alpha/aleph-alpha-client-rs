use aleph_alpha_client::{
    Client, Prompt, Sampling, SemanticRepresentation, TaskCompletion, TaskSemanticEmbedding, cosine_similarity,
};
use lazy_static::lazy_static;

lazy_static! {
    static ref AA_API_TOKEN: String = std::env::var("AA_API_TOKEN")
        .expect("AA_API_TOKEN environment variable must be specified to run tests.");
}

#[tokio::test]
async fn completion_with_luminous_base() {
    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello"),
        maximum_tokens: 1,
        sampling: Sampling::MOST_LIKELY,
    };

    let model = "luminous-base";

    let client = Client::new(&AA_API_TOKEN).unwrap();
    let response = client.execute(model, &task).await.unwrap();

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
    let query = Prompt::from_text("I am hungry, any idea what I could eat?");
    let client = Client::new(&AA_API_TOKEN).unwrap();
    let model = "luminous-base";

    // When
    let robot_embedding_task = TaskSemanticEmbedding {
        prompt: robot_fact,
        representation: SemanticRepresentation::Document,
        compress_to_size: Some(128),
    };
    let robot_embedding = client.execute(model, &robot_embedding_task).await.unwrap().embedding;

    let pizza_embedding_task = TaskSemanticEmbedding {
        prompt: pizza_fact,
        representation: SemanticRepresentation::Document,
        compress_to_size: Some(128),
    };
    let pizza_embedding = client.execute(model, &pizza_embedding_task).await.unwrap().embedding;

    let query_embedding_task = TaskSemanticEmbedding {
        prompt: query,
        representation: SemanticRepresentation::Query,
        compress_to_size: Some(128),
    };
    let query_embedding = client.execute(model, &query_embedding_task).await.unwrap().embedding;
    let similarity_pizza = cosine_similarity(&query_embedding, &pizza_embedding);
    println!("similarity pizza: {similarity_pizza}");
    let similarity_robot = cosine_similarity(&query_embedding, &robot_embedding);
    println!("similarity robot: {similarity_robot}");

    // Then

    // The fact about pizza should be more relevant to the "I'm hungry" question than a fact about
    // robots.
    assert!(similarity_pizza > similarity_robot);
}
