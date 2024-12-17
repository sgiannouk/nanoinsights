document.addEventListener("DOMContentLoaded", () => {
    const successAlert = document.querySelector('.alert-success');
    const errorAlert = document.querySelector('.alert-danger');

    if (successAlert) {
        // Reload the page after 5 seconds if success alert is present
        setTimeout(() => {
            window.location.href = successAlert.getAttribute('data-redirect-url');
        }, 5000);
    }

    if (errorAlert) {
        // Reload the page after 8 seconds if error alert is present
        setTimeout(() => {
            window.location.href = errorAlert.getAttribute('data-redirect-url');
        }, 8000);
    }
});
