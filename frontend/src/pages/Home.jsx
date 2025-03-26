import React, { useState } from "react";
import VirtualKeyboard from "../components/VirtualKeyboard";
import "./Home.css";

function Home() {
  const [searchTerm, setSearchTerm] = useState("");
  const [emotion, setEmotion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState("svm+countvectorizer");
  const [confidence, setConfidence] = useState(null);

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      setError("Please enter a Malayalam sentence.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: searchTerm,
          model: selectedModel,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch emotion prediction.");
      }

      const data = await response.json();
      setEmotion(data.emotion);
      setConfidence(data.confidence);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyClick = (key) => {
    setSearchTerm((prev) => prev + key);
  };

  return (
    <>
      <div className="container">
        <h1>Malayalam Emotion Detector</h1>
        <div>
          <input
            type="text"
            placeholder="Enter Malayalam sentence..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <div className="model-selector">
            <label htmlFor="model">Select Model: </label>
            <select
              id="model"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="countvectorizer+svm">CountVectorizer + SVM</option>
              <option value="countvectorizer+naivebayes">
                CountVectorizer + Naive Bayes
              </option>
              <option value="tf-idf+svm">TF-IDF + SVM </option>
              <option value="tf-idf+naivebayes">TF-IDF + Naive Bayes</option>
              <option value="bert+svm">BERT + SVM</option>
              <option value="bert+naivebayes">BERT + Naive Bayes</option>
              <option value="bert">BERT</option>
            </select>
          </div>
          <button onClick={handleSearch} disabled={loading}>
            {loading ? "Detecting..." : "Detect Emotion"}
          </button>
        </div>
        {error && <p style={{ color: "red" }}>{error}</p>}
        {emotion && (
          <div>
            <h2>Detected Emotion:</h2>
            <p>{emotion}</p>
            {confidence && (
              <p className="confidence-result">
                Confidence: {(confidence * 100).toFixed(2)}%
              </p>
            )}
          </div>
        )}
      </div>
      <VirtualKeyboard onKeyClick={handleKeyClick} />
    </>
  );
}

export default Home;
