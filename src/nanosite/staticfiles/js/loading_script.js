document.addEventListener("DOMContentLoaded", function () {
    const roadmap = document.getElementById("roadmap");
    const logContainer = document.getElementById("log-messages");

    function updateRoadmap(step) {
        // Reset all steps and lines
        roadmap.contentDocument.querySelectorAll(".step, .line").forEach((el) => {
            el.classList.remove("active");
        });

        // Highlight current step and connecting lines
        const currentStep = roadmap.contentDocument.getElementById(`step-${step}`);
        const currentLine = roadmap.contentDocument.getElementById(`line-${step - 1}-${step}`);

        if (currentStep) currentStep.classList.add("active");
        if (currentLine) currentLine.classList.add("active");
    }

    function fetchLogUpdates() {
        fetch("/get-log-updates/") // Backend endpoint
            .then((response) => response.json())
            .then((data) => {
                updateRoadmap(data.current_step);

                // Update log messages
                logContainer.innerHTML += `<p>${data.log_message}</p>`;
                logContainer.scrollTop = logContainer.scrollHeight; // Auto-scroll to the latest log
            });
    }

    // Fetch updates every 2 seconds
    setInterval(fetchLogUpdates, 2000);
});
