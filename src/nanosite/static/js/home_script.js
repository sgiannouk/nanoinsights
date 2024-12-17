document.addEventListener('DOMContentLoaded', function () {
    const message = document.getElementById('bottom-message');

    setTimeout(() => {
        message.style.opacity = '0'; // Start fading out
        setTimeout(() => {
            message.style.display = 'none'; // Fully remove from display
        }, 100); // Match the transition duration in CSS
    }, 10000); // 10000ms
});


//////////////////////////
window.onload = function () {
    const initialMessage = document.getElementById("initial-message");
    let hideTimeout; // Timer for hiding the message
    let hoverTimeout; // Separate timer for the 5-second disappearance after hover

    // Function to start the main hide timer (15 seconds)
    function startHideTimer() {
        hideTimeout = setTimeout(() => {
            fadeOutAndHide();
        }, 15000); // 15 seconds
    }

    // Function to stop the main hide timer
    function stopHideTimer() {
        clearTimeout(hideTimeout);
    }

    // Function to start the 5-second hover timer
    function startHoverTimer() {
        hoverTimeout = setTimeout(() => {
            fadeOutAndHide();
        }, 5000); // 5 seconds
    }

    // Function to stop the hover timer
    function stopHoverTimer() {
        clearTimeout(hoverTimeout);
    }

    // Function to fade out and hide the message
    function fadeOutAndHide() {
        initialMessage.style.transition = "opacity 0.5s ease";
        initialMessage.style.opacity = "0"; // Start fading out
        setTimeout(() => {
            initialMessage.style.display = "none"; // Hide completely
        }, 500); // Match the CSS transition duration (0.5 seconds)
    }

    // Show the message and start the hide timer
    function displayMessage() {
        if (!localStorage.getItem("hasSeenMessage")) {
            // Show the message
            initialMessage.style.display = "block";

            // Set the localStorage flag
            localStorage.setItem("hasSeenMessage", "true");

            // Start the hide timer
            startHideTimer();
        }
    }

    // Handle "dragging" as a simple click-to-dismiss event
    function makeDismissable() {
        initialMessage.addEventListener("mousedown", function () {
            fadeOutAndHide();
        });
    }

    // Event listeners for hover behavior
    initialMessage.addEventListener("mouseover", () => {
        stopHideTimer(); // Stop the main timer
        stopHoverTimer(); // Stop any existing hover timer
    });

    initialMessage.addEventListener("mouseleave", () => {
        startHoverTimer(); // Start the 5-second hover timer
    });

    // Initialize
    displayMessage();
    makeDismissable();
};
