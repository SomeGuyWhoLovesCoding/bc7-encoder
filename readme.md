### BC7 Encoder

This version actually knows what it's doing and optimizes for any real world texture with minimal compilation setup.

*So here's my story:*

So, Funkin' View needed another compressed texture format that's NATIVE to PC, and I naturally went to choose BC7 because all of the other pc-native texture compression algorithms just compress color or do basic block math.

However, in doing this, I encountered a few issues trying to find the most efficient bc7 encoder that's inside a command line tool and that you don't have to compile at all and use:

1. DirectXTex (made by Microsoft of all people)

...has a basic bc7/bptc encoder that's SLOW as hell. Doesn't help that it uses the GPU because too bad I'm on an IGPU which is a SHARED spot in the CPU, just with automatic SIMD running in there.

I don't give a shit about microsoft's products, even if I still don't mind windows 10 being end of support last year as of this repository's creation, so I'm not relying on that for real-time bc7 compression.

It utilizes no compute-saving optimizations, if at all, because it's stuck being an academic version made by a giant corp.

It's also wasting the GPU's lifespan, so why not just try to compile and then fuck it up for hours on end to be without that in the process?

...

Exactly.

2. Richgell's BC7E/F_RDO

You've never heard of this person, because he's had this strange and super-niche hobby of a specific texture compression known as bc7, where he tries to experiment with optimizing it, such as RDO for example (**Rate Distortion Optimization**), and even programming the whole ass encoder in ispc, which adds a few MORE layers of compelxity.

Fuck it, the ISPC version of bc7 compression was thankfully discontinued, and intel themselves have ceased all future updates of it.

When going to basis universal (which I actually found about very early on in my bc7 compression rabbithole), I saw it as a bulky ass compressed texture transcoder, which does not fit my purpose at all.

It does have bc7f in it, whichh

3. AMD's compressinator

I fucking hate how it performs. Compressinator uses the GPU as well, but it also feels like it's stuck as an ACADEMIC (literally) version, just like texconv.

It does have GPU options, but I'm also not willing to handle all gpu options, as they're technically useless anyway. It's like it's built for NVIDIA. Fuck compressinator, and I will always say fuck it.

4. My tool

After 3/4 of the journey though, I had the idea of getting AI to convert the richgell bc7enc's ispc file into actual c++ code that I can just work with, at all times.

I had tried getting the AI to write me a simd-to-width, which in fact didn't work at all, and just broke the AVX2 path with fine details of a specific type of block turning into a blocky mess, so I just removed that.

Second thing I tried testing was just having AI revamp my code to use SSE2 and then doing workstealing, only to get a glitchy looking texture that's consistent green-blue mess with way more alpha and random colors as THE 4x4 block.

Last thing I tried was just getting Qwen to suggest me some other optimizations, and I'll have to highlight the BEST of the ones I tried.

Basically:

1. I just reduced the max partitions for mode 2 from 64 to 8, which was already pretty impactful even doing 16, and then...

2. I just did a bit more small optimizations since I most often test spritesheets of a big ass size.

And for the last optimization I did, was just rewriting the `evaluate_solution` function to use integer LUT's, which let the compiler autovectorize the rest of that part, which is already impressive.



And for that, I've successfully crafted the most efficient multithreading bc7 encoder tool that's ready for you to just drop in your game and use in real time.

You can even load it in bulk safely as well, because you can essentially launch the same process over and over and it'd still work efficiently.

Also note there's no quality option because this is already at its max possible CPU efficiency whilst retaining visual fidelity.

*And no SIMD was even used in this (not even std::fma), as with what I said earlier. This library already compiles with autovectorization no matter what.*

I don't give a fuck if you credit me or not, because this tool would start showing up more often as it lives due to having great potential of being both beginner-friendly and optimized for extreme speed.

And as a final conclusion...

**This is the only bc7 compression tool that's both ultra-optimized and as a command-line tool, baked into my passion project "Funkin' View".**

The end. Now, please try it out. Your mind will be BLOWN.