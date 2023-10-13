// Get the validationType element
const validationType = document.getElementById("validationType");

// Get the extRawBox and extClinicalBox elements
const extRawBox = document.getElementById("extRawBox");
const extClinicalBox = document.getElementById("extClinicalBox");
const runBox = document.getElementById("runBox");

// Add an event listener to validationType
validationType.addEventListener("change", function () {
    // Get the selected option value
    const selectedOption = validationType.value;

    // Show or hide extRawBox, extClinicalBox and runBox based on the selected option
    if (selectedOption === "run") {
            runBox.style.display = "block";
            extRawBox.style.display = "none"; // Hide extRawBox
            extClinicalBox.style.display = "none"; // Hide extClinicalBox
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


// JavaScript to show/hide the "edgernormselection" section based on the "norm" dropdown
document.getElementById("norm").addEventListener("change", function() {
    var selectedValue = this.value;
    var edgernormSelection = document.getElementById("edgernormselection");

    if (selectedValue === "edger") {
        edgernormSelection.style.display = "block"; // Show the section
    } else {
        edgernormSelection.style.display = "none"; // Hide the section
    }
});


// Get references to the relevant elements
const featureSelectionDropdown = document.getElementById("featureselection");
const rfecrossvalField = document.getElementById("rfecrossvalopt");
const minfeatureField = document.getElementById("minfeatureopt");

// Function to show/hide the fields based on the selected value
function toggleFields() {
    if (featureSelectionDropdown.value === "RFE") {
        rfecrossvalField.style.display = "block";
        minfeatureField.style.display = "block";
    } else {
        rfecrossvalField.style.display = "none";
        minfeatureField.style.display = "none";
    }
}

// Add an event listener to the "featureselection" dropdown
featureSelectionDropdown.addEventListener("change", toggleFields);

// Initially, call the toggleFields function to set the initial state
toggleFields();





