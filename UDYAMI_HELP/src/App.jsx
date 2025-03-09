import { useState } from "react";

export default function App() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);

  const sendMessage = async () => {
    if (!message.trim()) return;
    setChat([...chat, { text: message, sender: "user" }]);
    setMessage("");
  
    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message }),
      });
  
      const data = await response.json();
  
      setChat((prev) => [...prev, { text: data.response, sender: "bot" }]);
    } catch (error) {
      console.error("Error:", error);
      setChat((prev) => [...prev, { text: "Error getting response.", sender: "bot" }]);
    }
  };
  

  return (
    <div style={styles.pageContainer}>
      <header style={styles.header}>UDYAMI HELPDESK</header>
      
      <div style={styles.chatContainer}>
        <div style={styles.chatWindow}>
          {chat.map((msg, index) => (
            <div
              key={index}
              style={{
                ...styles.message,
                alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
                backgroundColor: msg.sender === "user" ? "#007bff" : "#e0e0e0",
                color: msg.sender === "user" ? "#fff" : "#000",
              }}
            >
              {msg.text}
            </div>
          ))}
        </div>

        <div style={styles.inputContainer}>
          <input
            type="text"
            style={styles.input}
            placeholder="Type a message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button onClick={sendMessage} style={styles.button}>Send</button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  pageContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    backgroundColor: "#f4f4f9",
  },
  header: {
    fontSize: "24px",
    fontWeight: "bold",
    color: "#333",
    padding: "15px",
  },
  chatContainer: {
    width: "400px",
    backgroundColor: "#ffffff",
    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
    borderRadius: "10px",
    overflow: "hidden",
  },
  chatWindow: {
    height: "400px",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    padding: "15px",
    gap: "10px",
    backgroundColor: "#f9f9f9",
  },
  message: {
    maxWidth: "70%",
    padding: "10px",
    borderRadius: "8px",
  },
  inputContainer: {
    display: "flex",
    alignItems: "center",
    padding: "10px",
    backgroundColor: "#fff",
    borderTop: "1px solid #ddd",
  },
  input: {
    flex: 1,
    padding: "8px",
    border: "1px solid #ccc",
    borderRadius: "5px",
    outline: "none",
  },
  button: {
    marginLeft: "10px",
    backgroundColor: "#007bff",
    color: "white",
    padding: "8px 15px",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
};
