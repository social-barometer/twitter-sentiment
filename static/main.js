"use strict";

document.getElementById("analyse-tweet").addEventListener("submit", event => {
    event.preventDefault();
    const resultContainer = document.getElementById("results");
    const warning = document.getElementById("warning");
    const tweet = document.getElementById("tweet").value;
    const msg = {
        "tweets": [tweet]
    };

    warning.style.display = "block";
    resultContainer.innerHTML = "";

    fetch("/emotion-analysis", {
        body: JSON.stringify(msg),
        headers: {
            "content-type": "application/json"
        },
        method: "POST"
    })
        .then(resp => resp.json())
        .then(resp => {
            const analysis = resp[0];
            resultContainer.innerHTML = Object.keys(analysis).reduce((html, key) => {
                return html + `<li>${key}: ${analysis[key]}%</li>`;
            }, "");
            warning.style.display = "none";
        });
});
