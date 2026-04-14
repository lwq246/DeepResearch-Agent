from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient("localhost", port=6333)

# Define the filter
search_result = client.scroll(
    collection_name="arxiv_docs",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.title",
                match=models.MatchValue(value="The ATOM Report: Measuring the Open Language Model Ecosystem"),
            )
        ]
    ),
    limit=10 # Adjust based on how many chunks the paper has
)

# Access the points
points = search_result[0]
for point in points:
    print(point.payload['page_content'])