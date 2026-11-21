const input = document.querySelector("#image");
const button = document.querySelector("#segment");
const result = document.querySelector("#result");
const model = document.querySelector("#model");

button.addEventListener("click", async () => {
  if (!input.files.length) {
    result.textContent = "Choose an image first.";
    return;
  }
  const form = new FormData();
  form.append("file", input.files[0]);
  try {
    const response = await fetch(
      `http://localhost:8000/segment?model=${encodeURIComponent(model.value)}`,
      { method: "POST", body: form }
    );
    result.textContent = JSON.stringify(await response.json(), null, 2);
  } catch (error) {
    result.textContent = `Request failed: ${error.message}`;
  }
});
