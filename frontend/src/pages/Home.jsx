import React, { useState } from "react";
import VirtualKeyboard from "../components/VirtualKeyboard";

function Home() {
  const [searchTerm, setSearchTerm] = useState("");
  const [emotion, setEmotion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
        body: JSON.stringify({ text: searchTerm }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch emotion prediction.");
      }

      const data = await response.json();
      setEmotion(data.emotion);
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
          <button onClick={handleSearch} disabled={loading}>
            {loading ? "Detecting..." : "Detect Emotion"}
          </button>
        </div>
        {error && <p style={{ color: "red" }}>{error}</p>}
        {emotion && (
          <div>
            <h2>Detected Emotion:</h2>
            <p>{emotion}</p>
          </div>
        )}
      </div>
      <VirtualKeyboard onKeyClick={handleKeyClick} />
    </>
  );
}

export default Home;
