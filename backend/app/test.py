import base64
from colivara_py import ColiVara

api_key = "your-api-key"
pdf_path = "/Users/saipranavkrovvidi/Desktop/Machine-Learning-Algorithms-A-Review.pdf"  # or your actual file path
collection_name = "default_collection"
query = "How does the paper describe the concept and process of the k-nearest neighbors (KNN) algorithm, and what are its limitations in practical machine learning applications?"

client = ColiVara(api_key=api_key)

# Upload PDF as base64
with open(pdf_path, "rb") as f:
    encoded_content = base64.b64encode(f.read()).decode("utf-8")
doc = client.upsert_document(
    name="test.pdf",
    document_base64=encoded_content,
    collection_name=collection_name,
    wait=True
)
print("Upserted document:", doc)

# Search
search_response = client.search(query, collection_name=collection_name)
print("Raw search response:", search_response)
if hasattr(search_response, "results"):
    results = search_response.results
elif isinstance(search_response, tuple):
    results = search_response[0]
else:
    results = search_response

print("Results type:", type(results))
if results:
    print("First result type:", type(results[0]))
    print("First result dir:", dir(results[0]))
    print("First result dict:", getattr(results[0], '__dict__', None))
    print("First result repr:", repr(results[0]))
else:
    print("No results returned.")