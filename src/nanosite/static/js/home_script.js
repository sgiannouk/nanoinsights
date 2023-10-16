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


////////////////////////////

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


////////////////////////////

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


////////////////////////////

// Change the icon next to "Advanced Options" when you click on it
function toggleIcon() {
    const icon = document.getElementById('toggleIcon');
    if (icon.getAttribute('name') === 'chevron-down-outline') {
        icon.setAttribute('name', 'chevron-up-outline'); // Change to the desired icon name
    } else {
        icon.setAttribute('name', 'chevron-down-outline'); // Change back to the original icon
    }
}


////////////////////////////
// Uploading raw data
$('#dataUpload').bind('change', function () {
    var filename = $("#dataUpload").val();
    if (/^\s*$/.test(filename)) {
        $(".file-upload").removeClass('active');
        $("#noFile").text("No file chosen..."); 
    } else {
        $(".file-upload").addClass('active');
        $("#noFile").text(filename.replace("C:\\fakepath\\", "")); 
    }
});

$(document).ready(function () {
    $('#dataUpload').on('change', function () {
        var input = this;
        var files = input.files;

        if (files.length > 0) {
            // Files have been selected
            var text = files.length + (files.length === 1 ? ' file selected' : ' files selected');
            $('#noFile').text(text);
        } else {
            // No files chosen, reset the text to "No file chosen..."
            $('#noFile').text('No file chosen...');
        }
    });
});

// Uploading clinical data
$('#clinicalDataUpload').bind('change', function () {
    var filename = $("#clinicalDataUpload").val();
    if (/^\s*$/.test(filename)) {
        $(".file-upload").removeClass('active');
        $("#clinicalNoFile").text("No file chosen..."); 
    } else {
        $(".file-upload").addClass('active');
        $("#clinicalNoFile").text(filename.replace("C:\\fakepath\\", "")); 
    }
});

$(document).ready(function () {
    $('#clinicalDataUpload').on('change', function () {
        var input = this;
        var files = input.files;

        if (files.length > 0) {
            // Files have been selected
            var text = files.length === 1 ? files[0].name : files.length + ' files selected';
            $('#clinicalNoFile').text(text);
        } else {
            // No files chosen, reset the text to "No file chosen..."
            $('#clinicalNoFile').text('No file chosen...');
        }
    });
});
