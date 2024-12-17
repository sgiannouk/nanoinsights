document.addEventListener("DOMContentLoaded", function () {
    // Function to attach click handlers to thumbnails
    function attachThumbnailClickHandlers(tabId) {
        const tab = document.querySelector(`#${tabId}`);
        if (!tab) return; // If tab is not found, exit the function

        const wrappers = tab.querySelectorAll(".clickable-thumbnail-wrapper");
        const modal = document.querySelector(`#${tabId}Modal`);
        const modalIframe = modal ? modal.querySelector("iframe") : null;

        if (!modal || !modalIframe) {
            console.error(`Modal or iframe not found for tab: ${tabId}`);
            return;
        }

        wrappers.forEach((wrapper) => {
            wrapper.addEventListener("click", (event) => {
                event.preventDefault();
                const iframe = wrapper.querySelector("iframe");

                if (iframe) {
                    console.log(`Thumbnail clicked in tab: ${tabId}`);
                    modalIframe.src = iframe.getAttribute("src"); // Update iframe src
                    modal.classList.add("show");
                    modal.style.display = "flex";
                } else {
                    console.warn(`No iframe found in clicked thumbnail for tab: ${tabId}`);
                }
            });
        });

        // Close modal when clicking the close button
        const closeButton = modal.querySelector(".close");
        if (closeButton) {
            closeButton.addEventListener("click", () => {
                modal.classList.remove("show");
                modal.style.display = "none";
                modalIframe.src = ""; // Clear iframe source
                console.log(`Modal closed in tab: ${tabId}`);
            });
        }

        // Close modal when clicking outside the content
        modal.addEventListener("click", (event) => {
            if (event.target === modal) {
                modal.classList.remove("show");
                modal.style.display = "none";
                modalIframe.src = ""; // Clear iframe source
                console.log(`Clicked outside modal content in tab: ${tabId}`);
            }
        });
    }

    // Attach handlers to all tabs
    const tabs = document.querySelectorAll(".tab-pane");
    tabs.forEach((tab) => {
        const tabId = tab.getAttribute("id");
        attachThumbnailClickHandlers(tabId);
    });
});


document.addEventListener("DOMContentLoaded", function () {
    const downloadButton = document.getElementById("downloadResults");

    downloadButton.addEventListener("click", function (e) {
        e.preventDefault();

        // Get the project ID from the input or hidden field
        const projectId = document.getElementById("projectId").value;

        // Check if projectId exists and is valid
        if (!projectId) {
            alert("Project ID is missing. Please ensure the correct Project ID is provided.");
            return;
        }

        // Construct the file path dynamically
        const filePath = `/uploads/${projectId}/${projectId}_nanoinsights_results.zip`;

        // Create a temporary link and trigger the download
        const link = document.createElement("a");
        link.href = filePath;

        // Use the correct name for the downloaded file
        link.download = `${projectId}_nanoinsights_results.zip`;

        // Append the link to the body, trigger the click, and remove the link
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});

