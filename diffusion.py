from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1").to("cuda")

prompt = '风筝在空中飞舞，油画'
image = pipe(prompt, guidance_scale=7.5).images[0]  
image.save("飞流.png")