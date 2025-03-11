document.addEventListener("DOMContentLoaded", function () {
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-btn");

    // Function to send message
    function sendMessage() {
        let userMessage = userInput.value.trim();
        if (userMessage === "") return; // Prevent empty messages

        appendMessage("user", userMessage);
        userInput.value = ""; // Clear input field

        // Send user message to backend
        fetch("http://127.0.0.1:5000/chat", { 
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            formatAndAppendBotMessage(data.reply);
        })
        .catch(error => console.error("Error:", error));
    }

    // Click event for the Send button
    sendButton.addEventListener("click", sendMessage);

    // Keypress event for Enter key
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            event.preventDefault(); // Prevent form submission
            sendMessage();
        }
    });
});


// Function to append messages to the chat box
function appendMessage(sender, text) {
    let chatBox = document.getElementById("chat-box");
    let messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");

    // If bot response contains bullets (•), format them as a list
    if (sender === "bot" && text.includes("•")) {
        let list = document.createElement("ul");
        text.split("• ").forEach((item) => {
            if (item.trim() !== "") {
                let listItem = document.createElement("li");
                listItem.textContent = item.trim();
                list.appendChild(listItem);
            }
        });
        messageDiv.appendChild(list);
    } else {
        messageDiv.textContent = text;
    }

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to latest message
}


// Function to format and append bot response as bullet points
function formatAndAppendBotMessage(text) {
    let chatBox = document.getElementById("chat-box");
    let messageDiv = document.createElement("div");
    messageDiv.classList.add("message", "bot-message");

    // Convert response into bullet points if applicable
    let formattedResponse;
    if (text.includes("\n")) {
        formattedResponse = "<ul>" + text
            .split("\n") // Split by new lines
            .map(line => `<li>${line.replace(/^•\s?/, "").trim()}</li>`) // Remove existing bullets if present
            .join("") + "</ul>";
    } else {
        formattedResponse = `<p>${text}</p>`; // If single line, keep as paragraph
    }

    messageDiv.innerHTML = formattedResponse;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to latest message
}
