document.addEventListener('DOMContentLoaded', function () {
    // Select the sidebar and the burger button
    const sidebar = document.querySelector('.sidebar');
    const burgerButton = document.getElementById('btn');

    // Add a click event listener to toggle the "open" class
    burgerButton.addEventListener('click', function () {
        sidebar.classList.toggle('open'); // Toggle the "open" class
    });
});



////////////////////////////
// Event listener for training data uploads
function handleFileUpload(files, fieldName, uploadType = 'training') {
    const formData = new FormData();

    // Map variable names to user-friendly descriptions
    const variableAnnotations = {
        "raw_files[]": "Raw Data Files",
        "clinical_file": "Clinical Data File",
        "ext_raw_files[]": "External Raw Data Files",
        "ext_clinical_file": "External Clinical Data File"
    };

    // Resolve the user-friendly name for the field or fall back to the variable name
    const friendlyName = variableAnnotations[fieldName] || fieldName;

    // Add CSRF token
    formData.append('csrfmiddlewaretoken', getCsrfToken());

    // Append files to the FormData object
    if (fieldName.endsWith('raw_files[]')) {
        for (let i = 0; i < files.length; i++) {
            formData.append(fieldName, files[i]);
        }
    } else {
        formData.append(fieldName, files[0]); // Only one clinical file allowed
    }

    // Add upload type
    formData.append('upload_type', uploadType);

    // Send the files to the server
    fetch('/input_project/upload/', {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Success message with user-friendly annotation
                showNotification(`${friendlyName} uploaded successfully!`, true);
            } else {
                // Error message from the server
                showNotification(`Upload failed: ${data.message}`, false);
            }
        })
        .catch(error => {
            console.error('Error during file upload:', error);
            // Generic error message
            showNotification(`An error occurred while uploading ${friendlyName}.`, false);
        });
}

// Event Listeners for Training Data
document.getElementById('rawTrainingData').addEventListener('change', function () {
    handleFileUpload(this.files, 'raw_files[]', 'training');
});

document.getElementById('clinicalTrainingData').addEventListener('change', function () {
    handleFileUpload(this.files, 'clinical_file', 'training');
});

// Event Listeners for Validation Data
document.getElementById('rawValidationData').addEventListener('change', function () {
    handleFileUpload(this.files, 'ext_raw_files[]', 'validation');
});

document.getElementById('clinicalValidationData').addEventListener('change', function () {
    handleFileUpload(this.files, 'ext_clinical_file', 'validation');
});

function getCsrfToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    if (!token) {
        console.error("CSRF token not found in the DOM.");
        return null;
    }
    return token.value;
}




////////////////////////////
// Read the clinical data file and show the Control and Condition groups
document.getElementById('clinicalTrainingData').addEventListener('change', function () {
    const file = this.files[0];

    if (!file) {
        alert('No file selected.');
        return;
    }

    // Validate file type
    const validTypes = ['text/csv', 'text/plain'];
    if (!validTypes.includes(file.type)) {
        alert('Invalid file type. Please upload a CSV or TXT file.');
        return;
    }

    // Read the file
    const reader = new FileReader();
    reader.onload = function (event) {
        const fileContent = event.target.result;

        // Parse the file content
        const rows = fileContent.split('\n').map(row => row.trim());
        const header = rows[0].split(',');
        const conditionIndex = header.indexOf('Condition');

        if (conditionIndex === -1) {
            alert('The file must contain a "Condition" column.');
            return;
        }

        // Extract unique values from the "Condition" column
        const uniqueConditions = new Set();
        for (let i = 1; i < rows.length; i++) {
            const columns = rows[i].split(',');
            if (columns.length > conditionIndex) {
                uniqueConditions.add(columns[conditionIndex].trim());
            }
        }

        // Validate the number of unique conditions
        if (uniqueConditions.size !== 2) {
            alert('The "Condition" column must have exactly two unique values.');
            return;
        }

        // Populate the dropdowns
        const [condition1, condition2] = Array.from(uniqueConditions);
        const controlDropdown = document.getElementById('control');
        const conditionDropdown = document.getElementById('condition');

        // Clear existing options
        controlDropdown.innerHTML = `<option value="">Select Control</option>`;
        conditionDropdown.innerHTML = `<option value="">Select Condition</option>`;

        // Add new options
        controlDropdown.innerHTML += `
            <option value="${condition1}">${condition1}</option>
            <option value="${condition2}">${condition2}</option>
        `;
        conditionDropdown.innerHTML += `
            <option value="${condition1}">${condition1}</option>
            <option value="${condition2}">${condition2}</option>
        `;

        // Add event listeners to dynamically enable/disable options
        controlDropdown.addEventListener('change', function () {
            updateDropdownStates(controlDropdown, conditionDropdown, condition1, condition2);
        });

        conditionDropdown.addEventListener('change', function () {
            updateDropdownStates(conditionDropdown, controlDropdown, condition1, condition2);
        });
    };

    reader.onerror = function () {
        alert('Failed to read the file.');
    };

    reader.readAsText(file);
});

// Function to update dropdown states
function updateDropdownStates(changedDropdown, otherDropdown, condition1, condition2) {
    const selectedValue = changedDropdown.value;

    // Update the other dropdown's options
    Array.from(otherDropdown.options).forEach(option => {
        if (option.value === selectedValue) {
            option.disabled = true;
        } else {
            option.disabled = false;
        }
    });
}




////////////////////////////
// Get the testingType element
const testingType = document.getElementById("testingType");

// Get the extRawBox, extClinicalBox, and runBox elements
const extRawBox = document.getElementById("extRawBox");
const extClinicalBox = document.getElementById("extClinicalBox");
const runBox = document.getElementById("runBox");
const runSelect = document.getElementById("run");

// Add an event listener to testingType
testingType.addEventListener("change", function () {
    const selectedOption = testingType.value;
    const clinicalFileInput = document.getElementById("clinicalTrainingData");

    if (selectedOption === "run") {
        if (!clinicalFileInput.files.length) {
            showNotification("Please upload the NanoString Raw Data Files before selecting 'Runs' as the validation type.", false);
            testingType.value = "split"; // Reset to a default value
            return;
        }
        runBox.style.display = "block";
        extRawBox.style.display = "none"; // Hide extRawBox
        extClinicalBox.style.display = "none"; // Hide extClinicalBox

        // Fetch CartridgeIDs from the backend
        fetchCartridgeIDs();
    } else if (selectedOption === "extSet") {
        extRawBox.style.display = "block";
        extClinicalBox.style.display = "block";
        runBox.style.display = "none"; // Hide runBox
    } else {
        runBox.style.display = "none";
        extRawBox.style.display = "none";
        extClinicalBox.style.display = "none";
    }
});


// Function to fetch CartridgeIDs from the backend
function fetchCartridgeIDs() {
    const projectId = getProjectIdFromSession(); // Retrieve the project ID from the session or context
    if (!projectId) {
        alert("Project ID is missing. Ensure you have uploaded files.");
        return;
    }

    fetch(`/input_project/get_cartridge_ids/?project_id=${projectId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                updateRunBox(new Set(data.cartridge_ids));
            } else {
                alert(`Error fetching CartridgeIDs: ${data.message}`);
            }
        })
        .catch(error => {
            console.error("Error fetching CartridgeIDs:", error);
            alert("An error occurred while fetching CartridgeIDs.");
        });
}

// Function to update the runBox dropdown
function updateRunBox(cartridgeIDs) {
    runSelect.innerHTML = ""; // Clear previous options

    cartridgeIDs.forEach(id => {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        runSelect.appendChild(option);
    });
}

// Utility function to retrieve the project ID from the session
function getProjectIdFromSession() {
    const projectInput = document.getElementById("projectId");
    if (!projectInput || !projectInput.value) {
        alert("Project ID is missing. Ensure you have uploaded files.");
        return null;
    }
    return projectInput.value;
}




////////////////////////////
// Hide/Show rfecrossvalField and minfeatureField based on featureSelectionDropdown
const featureSelectionDropdown = document.getElementById("featureselection");
const rfecrossvalField = document.getElementById("rfecrossvalopt");
const minfeatureField = document.getElementById("minfeatureopt");

function toggleFields() {
    if (featureSelectionDropdown.value === "RFE") {
        rfecrossvalField.style.display = "block";
        minfeatureField.style.display = "block";
    } else if (featureSelectionDropdown.value === 'PI') {
        rfecrossvalField.style.display = "none";
    } else {
        rfecrossvalField.style.display = "none";
        minfeatureField.style.display = "none";
    }
}
// Add an event listener to the "featureselection" dropdown
featureSelectionDropdown.addEventListener("change", toggleFields);

// Initially, call the toggleFields function to set the initial state
toggleFields();



////////////////////////////
// Collapse the Advanced Options menus
document.getElementById("toggleAdvancedOptions").addEventListener("click", function() {
    var advancedOptions = document.getElementById("collapseadvancedoptions");
    if (advancedOptions.classList.contains("show")) {
        advancedOptions.classList.remove("show");
    } else {
        advancedOptions.classList.add("show");
    }
});

document.addEventListener("DOMContentLoaded", function () {
    var toggleButton = document.getElementById("toggleAdvancedOptions");
    var collapseElement = document.getElementById("collapseadvancedoptions");

    toggleButton.addEventListener("click", function () {
        if (collapseElement.classList.contains("show")) {
            // Collapse is shown; scroll to the bottom
            collapseElement.scrollIntoView({ behavior: "smooth" });
        } else {
            // Collapse is hidden; scroll to the top
            toggleButton.scrollIntoView({ behavior: "smooth" });
        }
    });
});



// Collapse the Filter Options menus
document.getElementById("toggleFilterOptions").addEventListener("click", function() {
    var filterOptions = document.getElementById("collapsefilteroptions");
    if (filterOptions.classList.contains("show")) {
        filterOptions.classList.remove("show");
    } else {
        filterOptions.classList.add("show");
    }
});

document.addEventListener("DOMContentLoaded", function () {
    var toggleButton = document.getElementById("toggleFilterOptions");
    var collapseElement = document.getElementById("collapsefilterdoptions");

    toggleButton.addEventListener("click", function () {
        if (collapseElement.classList.contains("show")) {
            // Collapse is shown; scroll to the bottom
            collapseElement.scrollIntoView({ behavior: "smooth" });
        } else {
            // Collapse is hidden; scroll to the top
            toggleButton.scrollIntoView({ behavior: "smooth" });
        }
    });
});



////////////////////////////
// Change the icon next to "Advanced Options" and "Filters" when you click on it
function toggleIcon() {
    const icon = document.getElementById('toggleIcon');
    if (icon.getAttribute('name') === 'chevron-down-outline') {
        icon.setAttribute('name', 'chevron-up-outline'); // Change to the desired icon name
    } else {
        icon.setAttribute('name', 'chevron-down-outline'); // Change back to the original icon
    }
}



////////////////////////////
// Submit button
let logPollInterval = null; // Store polling interval globally

document.getElementById('button').addEventListener('click', function (event) {
    event.preventDefault(); // Prevent default form submission

    const projectId = document.getElementById('projectId').value; // Retrieve the project ID from the hidden input
    const formData = new FormData();
    const missingFields = [];

    // Show Loader
    showLoader();

    // Map variable names to user-friendly descriptions
    const variableAnnotations = {
        "raw_files[]": "Raw Data Files",
        "clinical_file": "Clinical Data File",
        "control": "Control Group",
        "condition": "Condition Group",
        "run[]": "Runs"
    };

    // Check required fields
    if (document.querySelector('input[name="raw_files[]"]').files.length === 0) {
        missingFields.push(variableAnnotations["raw_files[]"]);
    }
    if (document.querySelector('input[name="clinical_file"]').files.length === 0) {
        missingFields.push(variableAnnotations["clinical_file"]);
    }
    if (document.querySelector('select[name="control"]').value === '') {
        missingFields.push(variableAnnotations["control"]);
    }
    if (document.querySelector('select[name="condition"]').value === '') {
        missingFields.push(variableAnnotations["condition"]);
    }

    // Check for "Runs" validation if testingType is "run"
    const testingType = document.getElementById('testingType').value;
    if (testingType === 'run') {
        const selectedRuns = document.querySelector('select[name="run[]"]').selectedOptions;
        if (selectedRuns.length === 0) {
            missingFields.push("At least one run should be chosen");
        }
    }

    if (missingFields.length > 0) {
        const message = missingFields.length === 1
            ? `The following field is missing: ${missingFields[0]}.\nPlease complete it before submitting!`
            : `The following fields are missing: ${missingFields.join(", ")}.\nPlease complete them before submitting!`;

        showNotification(message, false);
        hideLoader();
        return; // Stop further execution
    }

    // Collect inputs
    document.querySelectorAll('select[multiple], .container-left input, .container-left select, .container-right input, .container-right select').forEach(input => {
        if (input.type === 'file') {
            Array.from(input.files).forEach(file => formData.append(input.name, file));
        } else if (input.type === 'checkbox' || input.type === 'radio') {
            if (input.checked) formData.append(input.name, input.value);
        } else if (!formData.has(input.name)) {
            formData.append(input.name, input.value);
        }
    });

    formData.append('submit', 'true'); // Add submit indicator

    // Disable the submit button
    const submitButton = document.getElementById('button');
    submitButton.disabled = true;

    // Step 1: Upload data
    fetch('/input_project/upload/', {
        method: 'POST',
        body: formData,
        headers: { 'X-CSRFToken': getCsrfToken() },
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showNotification('Files uploaded and configuration saved!', true);

            // Step 2: Trigger NanoInsights backend script
            return fetch(`/input_project/run-nanoinsights/${projectId}/`, {
                method: 'POST',
                headers: { 'X-CSRFToken': getCsrfToken() },
            });
        } else {
            throw new Error(data.message || 'Upload failed');
        }
    })
    .then(runResponse => runResponse.json())
    .then(runData => {
        if (runData.success) {
            showNotification('NanoInsights script started successfully!', true);

            // Step 3: Start polling the log file
            pollLogFile(projectId);
        } else {
            throw new Error(runData.error || 'NanoInsights script failed.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification(`Error: ${error.message}`, false);
        hideLoader();
    });
});

function pollLogFile(projectId) {
    const logFileUrl = `/input_project/log/${projectId}/`; // Corrected path
    const myAnalysisTab = document.getElementById('myAnalysis');

    logPollInterval = setInterval(() => {
        fetch(logFileUrl)
            .then(response => {
                if (!response.ok) throw new Error("Log file not ready yet");
                return response.text();
            })
            .then(data => {
                const lines = data.trim().split('\n');
                const lastMessage = JSON.parse(lines[lines.length - 1]).message;

                if (lastMessage.includes("ERROR")) {
                    clearInterval(logPollInterval);
                    showNotification(`Error: ${lastMessage}`, false);
                    hideLoader();
                } else if (lastMessage.includes("### FINISHED SUCCESSFULLY ###")) {
                    clearInterval(logPollInterval);
                    showNotification('NanoInsights analysis completed successfully!', true);
                    hideLoader();

                    // Reactivate My Analysis tab
                    if (myAnalysisTab) {
                        myAnalysisTab.href = `/input_project/analysis/${projectId}/`;
                        myAnalysisTab.classList.remove('inactive', 'disabled');
                        myAnalysisTab.style.pointerEvents = 'auto';
                        myAnalysisTab.style.color = ''; // Restore default color
                        myAnalysisTab.title = "Go to your analysis results";

                        // Store project ID in localStorage
                        localStorage.setItem('currentProjectId', projectId);

                        // Add click listener
                        myAnalysisTab.addEventListener('click', () => {
                            window.location.href = `/input_project/analysis/${projectId}/`;
                        });
                    }

                    // Redirect to the analysis page automatically
                    window.location.href = `/input_project/analysis/${projectId}/`;
                }
            })
            .catch(error => {
                console.log('Waiting for log file...', error.message);
            });
    }, 5000); // Poll every 5 seconds
}

document.addEventListener('DOMContentLoaded', function () {
    const myAnalysisTab = document.getElementById('myAnalysis');
    const projectId = localStorage.getItem('currentProjectId');

    if (projectId && myAnalysisTab) {
        myAnalysisTab.href = `/input_project/analysis/${projectId}/`;
        myAnalysisTab.classList.remove('inactive', 'disabled');
        myAnalysisTab.style.pointerEvents = 'auto';
        myAnalysisTab.style.opacity = '1';
        myAnalysisTab.style.color = ''; // Restore default text color
        myAnalysisTab.title = "Go to your analysis results";

        // Restore hover behavior dynamically
        myAnalysisTab.addEventListener('mouseover', () => {
            myAnalysisTab.style.backgroundColor = 'var(--color-text)';
            myAnalysisTab.style.color = 'var(--color-white)';
        });

        myAnalysisTab.addEventListener('mouseout', () => {
            myAnalysisTab.style.backgroundColor = '';
            myAnalysisTab.style.color = 'var(--color-text)';
        });
    }
});

function showLoader() {
    const loader = document.getElementById('loader');
    if (loader) loader.style.display = 'flex';
}

function hideLoader() {
    const loader = document.getElementById('loader');
    if (loader) loader.style.display = 'none';
    if (logPollInterval) clearInterval(logPollInterval);
}

function getCsrfToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    return token ? token.value : '';
}

function showNotification(message, isSuccess) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification ${isSuccess ? 'success' : 'error'}`;
    notification.innerText = message;

    container.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('fadeOut');
        setTimeout(() => notification.remove(), 1000);
    }, isSuccess ? 3000 : 6000);
}







////////////////////////////
// Helper function to show notifications
function showNotification(message, isSuccess) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification ${isSuccess ? 'success' : 'error'}`;
    notification.innerText = message;

    container.appendChild(notification);

    // Set display duration: 3 seconds for success, 6 seconds for error
    const displayDuration = isSuccess ? 3000 : 6000;

    setTimeout(() => {
        notification.classList.add('fadeOut');
        setTimeout(() => notification.remove(), 1000); // Wait for animation to finish
    }, displayDuration);
}

////////////////////////////
// Helper function to get CSRF token
function getCsrfToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    if (!token) {
        console.error("CSRF token not found in the DOM.");
        return null;
    }
    return token.value;
}





//////////////////////
window.addEventListener('beforeunload', function () {
    const projectId = document.getElementById('projectId')?.value;
    if (projectId) {
        const configPathExists = document.querySelector('#configExists')?.value === 'true';
        if (!configPathExists) {
            navigator.sendBeacon('/delete_project/', JSON.stringify({ project_id: projectId }));
        }
    }
});



//////////////////////
// Functions for Notifications
function showNotification(message, isSuccess) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification ${isSuccess ? 'success' : 'error'}`;
    notification.innerText = message;

    container.appendChild(notification);

    // Set display duration: 3 seconds for success, 8 seconds for error
    const displayDuration = isSuccess ? 3000 : 6000;

    setTimeout(() => {
        notification.classList.add('fadeOut');
        setTimeout(() => notification.remove(), 1000); // Wait for animation to finish
    }, displayDuration);
}


document.addEventListener("DOMContentLoaded", function () {
    const radioOption1 = document.getElementById("radioOption1");
    const radioOption2 = document.getElementById("radioOption2");

    // Ensure radioOption1 is selected
    radioOption1.checked = true;

    // Disable and uncheck radioOption2
    radioOption2.disabled = true;
    radioOption2.checked = false;
});


document.addEventListener("DOMContentLoaded", function () {
    // Control dropdown
    const controlSelect = document.getElementById("control");
    controlSelect.innerHTML = `
        <option value="negative" selected>negative</option>
    `;
    controlSelect.disabled = true; // Make it unchangeable if needed

    // Condition dropdown
    const conditionSelect = document.getElementById("condition");
    conditionSelect.innerHTML = `
        <option value="positive" selected>positive</option>
    `;
    conditionSelect.disabled = true; // Make it unchangeable if needed
});



document.addEventListener("DOMContentLoaded", function () {
    // Paths to preloaded example files
    const exampleRawData = "GSE81983_RAW.tar.gz";
    const exampleClinicalData = "clinical_data.csv";

    // Raw Training Data Input
    const rawTrainingDataInput = document.getElementById("rawTrainingData");
    const rawFileContainer = document.createElement("p");
    rawFileContainer.textContent = `Example File: ${exampleRawData}`;
    rawFileContainer.style.color = "green";
    rawTrainingDataInput.insertAdjacentElement("afterend", rawFileContainer);

    // Disable the input field to simulate file lock
    rawTrainingDataInput.disabled = true;

    // Clinical Data Input
    const clinicalTrainingDataInput = document.getElementById("clinicalTrainingData");
    const clinicalFileContainer = document.createElement("p");
    clinicalFileContainer.textContent = `Example File: ${exampleClinicalData}`;
    clinicalFileContainer.style.color = "green";
    clinicalTrainingDataInput.insertAdjacentElement("afterend", clinicalFileContainer);

    // Disable the input field to simulate file lock
    clinicalTrainingDataInput.disabled = true;
});


document.addEventListener('DOMContentLoaded', function () {
    const projectId = document.getElementById('projectId').value;

    // Trigger automatic analysis polling
    if (projectId) {
        pollLogFile(projectId);
    }
});