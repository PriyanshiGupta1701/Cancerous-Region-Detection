document.getElementById('upload-button').addEventListener('click', () => {
    document.getElementById('file-input').click();
  });
  
  document.getElementById('file-input').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;
  
    const formData = new FormData();
    formData.append('image', file);
  
    try {
      const response = await fetch('/', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) throw new Error('Upload failed');
  
      const data = await response.json();
  
      document.getElementById('upload-container').style.display = 'none';
      const inputImageContainer = document.getElementById('input-image-container');
      inputImageContainer.style.display = 'block';
      document.getElementById('input-image').src = data.input_image + '?t=' + new Date().getTime();
      document.getElementById('file-name').textContent = file.name;
  
      // Hide output container and spinner on new upload
      document.getElementById('output-container').style.display = 'none';
      document.getElementById('spinner').style.display = 'none';
    } catch (error) {
      alert('Failed to upload image. Please try again.');
    }
  });
  
  document.getElementById('delete-button').addEventListener('click', () => {
    document.getElementById('input-image-container').style.display = 'none';
    document.getElementById('upload-container').style.display = 'block';
  
    // Clear image src and file name
    document.getElementById('input-image').src = '';
    document.getElementById('file-name').textContent = '';
  
    // Hide output container and spinner
    document.getElementById('output-container').style.display = 'none';
    document.getElementById('spinner').style.display = 'none';
  
    // Clear file input value
    document.getElementById('file-input').value = '';
  });
  
  document.getElementById('detect-button').addEventListener('click', async () => {
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('output-container').style.display = 'none';
  
    try {
      const response = await fetch('/detect', { method: 'POST' });
      if (!response.ok) throw new Error('Detection failed');
  
      const data = await response.json();
  
      if (data.error) {
        alert(data.error);
        return;
      }
  
      document.getElementById('detectron-output').src = data.detection_image + '?t=' + new Date().getTime();
      document.getElementById('sam-output').src = data.segmented_image + '?t=' + new Date().getTime();
      document.getElementById('output-container').style.display = 'flex';
    } catch (error) {
      alert('Error processing image.');
    } finally {
      document.getElementById('spinner').style.display = 'none';
    }
  });