function drawWave(canvas) {
  const ctx = canvas.getContext('2d');
  const amplitude = 50;
  const frequency = 0.02;
  const speed = 0.1;

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();

    for (let x = 0; x < canvas.width; x += 5) {
      const y = amplitude * Math.sin(frequency * x + speed * Date.now() * 0.001);
      ctx.lineTo(x, canvas.height / 2 + y);
    }

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#3498db';
    ctx.stroke();

    requestAnimationFrame(animate);
  }

  animate();
}
