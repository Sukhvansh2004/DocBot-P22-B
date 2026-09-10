function sendQuery() {
    var query = document.getElementById('query').value.trim();
    if (query === '') {
        document.getElementById('query').value = '';
        return;
    }

    showLoadingBar();

    // Display the user's input as a chat message
    var chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML += '<div class="user-message">' + query + '</div>';
    chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom

    fetch('/send?query=' + encodeURIComponent(query))
        .then(response => {
            const reader = response.body.getReader();
            return new ReadableStream({
                start(controller) {
                    function push() {
                        reader.read().then(({ done, value }) => {
                            if (done) {
                                controller.close();
                                hideLoadingBar();
                                document.getElementById('query').value = ''; // Clear input box
                                return;
                            }
                            const line = new TextDecoder().decode(value).trimEnd();
                            if (line.startsWith('Progress:')) {
                                const [_, progress] = line.split(': ');
                                updateLoadingBar(progress);
                            } else if (line.endsWith('.pdf')) {
                                const pdfPath = line; // Assuming the response contains the full PDF path
                                chatMessages.innerHTML += `<div class="response pdf">` + pdfPath + `</div>`;
                                chatMessages.innerHTML += `<div class="response pdf"><iframe src="${pdfPath}" style="width: 100%; height: 500px;"></iframe></div>`;
                                chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom
                                printJS(pdfPath, 'pdf');
                            } else {
                                chatMessages.innerHTML += '<div class="response"><strong>Response:</strong>' + line + '</div>';
                                chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom
                            }
                            push();
                        });
                    }
                    push();
                }
            });
        })
        .then(stream => {
            const reader = stream.getReader();
            return new ReadableStream({
                start(controller) {
                    function push() {
                        reader.read().then(({ done, value }) => {
                            if (done) {
                                controller.close();
                                return;
                            }
                            controller.enqueue(value);
                            push();
                        });
                    }
                    push();
                }
            });
        })
        .then(stream => new Response(stream).text())
        .then(text => {
            console.log(text);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('query').value = ''; // Clear input box even on error
        });
}

// ... (the rest of the code remains the same) ...

function formatInput(input) {
    const maxCharsPerLine = 28; // Change this value to adjust the maximum number of characters per line
    let formattedInput = '';
    let currentLine = '';

    for (let i = 0; i < input.length; i++) {
        const char = input[i];

        if (char === '\n' || currentLine.length === maxCharsPerLine) {
            formattedInput += currentLine + '\n';
            currentLine = '';
        }

        if (char !== '\n') {
            currentLine += char;
        }
    }

    formattedInput += currentLine;
    return formattedInput;
}

function updateLoadingBar(progress) {
    const [current, total] = progress.split('/');
    const percentage = (current / total) * 100;
    var progressBar = document.getElementById('progressBar');
    progressBar.style.width = `${percentage}%`;
    progressBar.innerHTML = `${percentage.toFixed(0)}%`;
}

function showLoadingBar() {
    updateLoadingBar('0/100');
}

function hideLoadingBar() {
    updateLoadingBar('100/100');
    setTimeout(() => {
        updateLoadingBar('0/100');
    }, 1000);
}

const enButton = document.querySelector('button[onclick="EN()"]');
let isEnglish = true;

function toggleLanguage() {
  if (isEnglish) {
    enButton.textContent = 'JP';
    isEnglish = false;
    fetchLanguageToggle('jp');
  } else {
    enButton.textContent = 'EN';
    isEnglish = true;
    fetchLanguageToggle('en');
  }
}

function fetchLanguageToggle(language) {
  fetch('/toggle_language', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ language: language })
  })
  .then(response => response.text())
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));
}

enButton.addEventListener('click', toggleLanguage);

function clearChat() {
    document.getElementById('chat-messages').innerHTML = '';
}

function uploadPDF() {
    var pdfFile = document.getElementById('pdfFile').files[0];
    if (pdfFile) {
        showLoadingBar();
        var formData = new FormData();
        formData.append('pdfFile', pdfFile);
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(text => {
            console.log(text);
            // Display the uploaded PDF in the viewer
            var pdfViewer = document.getElementById('pdf-viewer');
            pdfViewer.innerHTML = `<iframe src="${URL.createObjectURL(pdfFile)}" width="100%" height="100%" style="border: none;"></iframe>`;
            hideLoadingBar();
        });
    }
}

const queryInput = document.getElementById('query');
const maxCharsPercentage = 0.85; // Set the maximum width percentage

queryInput.addEventListener('input', () => {
  const containerWidth = inputContainer.offsetWidth;
  const maxChars = Math.floor(containerWidth * 0.85); // 85% of the container width

  if (queryInput.value.length >= maxChars) {
    queryInput.style.width = `${containerWidth * 0.85}px`;
    queryInput.style.wordWrap = 'break-word';
    queryInput.classList.add('scrollable'); // Add the scrollable class
  } else {
    queryInput.style.width = `${containerWidth}px`;
    queryInput.style.wordWrap = 'initial';
    queryInput.classList.remove('scrollable'); // Remove the scrollable class
  }
});
queryInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        sendQuery();
        queryInput.value = ''; // Clear the input field after sending the query
    }
});


// Listen for the 'beforeunload' event
window.addEventListener('beforeunload', function() {
    // Send a request to the Flask server to delete the .ann file
    fetch('/delete_ann_file', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log('Received response from server:', response.status);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});