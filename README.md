# CAPSTONE_PROJECT
Capstone Project HCMUT - Agentic Search

## 🧠 Image Query Service

This service processes user queries with optional image inputs and filters.  
It follows a strict JSON schema for requests and responses.

---

### 📥 Request Format
```json
{
  "user_id": "user123",
  "session_id": "sess_45b2",
  "query": "Detect all objects in this image",
  "image": {
    "image_url": "https://example.com/cat.png",
    "image_base64": null,
    "image_type": "png"
  },
  "filter": {
    "video_url": [],
    "timestamp": null
  }
}
```


### 📥 Response Format
```json
{
  "response": "Detected objects: cat, sofa, plant",
  "error": null,
  "image": {
    "image_url": "https://example.com/annotated_cat.png",
    "image_base64": null,
    "image_type": "png"
  }
}
```
