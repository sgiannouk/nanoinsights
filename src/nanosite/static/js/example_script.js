document.addEventListener("DOMContentLoaded", function () {
    const prefilledData = JSON.parse(document.getElementById("prefilled-data").textContent);

    for (const [key, value] of Object.entries(prefilledData)) {
        const element = document.querySelector(`[name="${key}"]`);
        if (element) {
            if (element.type === "file") {
                // Simulate file upload display
                const fileLabel = document.createElement("span");
                fileLabel.textContent = `Example File: ${value.split('/').pop()}`;
                element.insertAdjacentElement("afterend", fileLabel);
            } else if (element.type === "select-one" || element.type === "select-multiple") {
                Array.from(element.options).forEach(option => {
                    if (option.value === value) {
                        option.selected = true;
                    }
                });
            } else {
                element.value = value;
            }
        }
    }
});
