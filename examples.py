"""
Example usage scripts for the Revenue Intelligence Platform
"""

# Example 1: Complete Setup and Training
def example_full_setup():
    """Run complete setup from scratch"""
    import subprocess
    
    # Run full setup
    subprocess.run(['python', 'main.py', '--steps', 'all'])
    
    # Start API server
    subprocess.run(['python', '-m', 'api.server'])


# Example 2: Make Predictions
def example_predictions():
    """Make forecasts using trained models"""
    from forecasting.predict import ForecastingPredictor
    import pandas as pd
    
    # Load data
    data = pd.read_csv('data/processed/train_processed.csv')
    
    # Create predictor
    predictor = ForecastingPredictor(model_type='lstm')
    
    # Predict for Store 1, Department 1
    result = predictor.predict_store(
        data,
        store_id=1,
        dept_id=1,
        horizon=4
    )
    
    print(f"Predictions: {result['predictions']}")
    print(f"Dates: {result['forecast_dates']}")


# Example 3: Query Neo4j
def example_neo4j_query():
    """Query Neo4j knowledge graph"""
    from knowledge_graph.neo4j_query import Neo4jQuery
    
    query = Neo4jQuery()
    
    # Get store insights
    insights = query.get_store_insights(store_id=1)
    
    print(f"Store Info: {insights['store_info']}")
    print(f"Top Departments: {insights['departments'][:3]}")
    print(f"Similar Stores: {insights['similar_stores']}")
    
    query.close()


# Example 4: RAG Question Answering
def example_rag_qa():
    """Ask questions using RAG"""
    from rag.rag_pipeline import RAGPipeline
    
    pipeline = RAGPipeline(use_neo4j=True, use_hybrid_retrieval=True)
    
    questions = [
        "What are the top performing stores?",
        "How do holiday sales compare to regular sales?",
        "Which departments have the highest sales?"
    ]
    
    for question in questions:
        result = pipeline.answer_question(question, store_id=1)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
    
    pipeline.close()


# Example 5: API Client
def example_api_client():
    """Use the API programmatically"""
    import requests
    
    base_url = "http://localhost:8000"
    
    # Forecast
    response = requests.post(f"{base_url}/forecast", json={
        "store_id": 1,
        "dept_id": 1,
        "horizon": 4,
        "model_type": "transformer",
        "natural_language_query": "Explain the forecast"
    })
    
    forecast = response.json()
    print(f"Forecast: {forecast['predictions']}")
    print(f"Explanation: {forecast['explanation']}")
    
    # RAG Answer
    response = requests.post(f"{base_url}/rag-answer", json={
        "question": "What drives sales in Store 1?",
        "store_id": 1
    })
    
    answer = response.json()
    print(f"\nAnswer: {answer['answer']}")
    
    # Graph Insights
    response = requests.post(f"{base_url}/graph-insights", json={
        "store_id": 1,
        "include_similar_stores": True
    })
    
    insights = response.json()
    print(f"\nSimilar Stores: {insights['similar_stores']}")


# Example 6: Batch Predictions
def example_batch_predictions():
    """Make predictions for multiple stores"""
    from forecasting.predict import ForecastingPredictor
    import pandas as pd
    
    data = pd.read_csv('data/processed/train_processed.csv')
    predictor = ForecastingPredictor(model_type='lstm')
    
    # Batch predict for stores 1-5
    results = predictor.batch_predict(
        data,
        store_ids=[1, 2, 3, 4, 5],
        horizon=4
    )
    
    for result in results:
        print(f"Store {result['store_id']}, Dept {result['dept_id']}: "
              f"{result['predictions']}")


if __name__ == "__main__":
    print("Revenue Intelligence Platform - Examples")
    print("=" * 50)
    print("\nAvailable examples:")
    print("1. example_full_setup() - Complete setup")
    print("2. example_predictions() - Make forecasts")
    print("3. example_neo4j_query() - Query knowledge graph")
    print("4. example_rag_qa() - Ask questions")
    print("5. example_api_client() - Use API")
    print("6. example_batch_predictions() - Batch forecasts")
    print("\nRun any function to see it in action!")
