const input = document.querySelector("#image");
const button = document.querySelector("#predict");
const result = document.querySelector("#result");

button.addEventListener("click", async () => {
  if (!input.files.length) {
    result.textContent = "Choose an image first.";
    return;
  }
  const form = new FormData();
  form.append("file", input.files[0]);
  const response = await fetch("http://localhost:8000/predict", { method: "POST", body: form });
  result.textContent = JSON.stringify(await response.json(), null, 2);
});
