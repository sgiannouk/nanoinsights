function generateBalls() {
    for (var i = 0; i < Math.floor(window.innerWidth); i++) {
    $(".gooey-animations").append(`<div class="ball"></div>`);
    var colors = ["#C2BCE6", "#F0ECCA", "#E2C1E4", "#CAEBE9", "#F0CECE", "DBF5D5"];
    // var colors = ["#C2DED1", "#99C5B5", "#FFE6E6", "#C6E2E9", "#8FBDD3"];
    $(".ball")
        .eq(i)
        .css({
        bottom: "-200px",
        left: Math.random() * window.innerWidth - 180,
        "animation-delay": Math.random() * 20 + "s",
        transform: "translateY(" + Math.random() * 1500 + "px)",
        "background-color": colors[i % colors.length]
        });
    }
}

generateBalls();

    window.addEventListener("resize", function (e) {
        $(".gooey-animations .ball").remove();
        generateBalls();
    });