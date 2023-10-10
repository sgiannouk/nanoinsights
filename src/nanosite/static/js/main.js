document.addEventListener('DOMContentLoaded', function () {
    const sidebarCollapseButton = document.getElementById('sidebarCollapse');
    const sidebar = document.querySelector('.sidebar');

    sidebarCollapseButton.addEventListener('click', function () {
        sidebar.classList.toggle('collapsed');
    });
});