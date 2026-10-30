/* UI-only polish. No metric, threshold or inference state is transformed. */
(()=>{'use strict';
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const paused=()=>document.body.classList.contains('animations-paused');
 const sliders=[...document.querySelectorAll('input[type="range"]')];
 const paint=el=>{const min=Number(el.min)||0,max=Number(el.max)||100;el.style.setProperty('--range-progress',`${Math.max(0,Math.min(100,(Number(el.value)-min)/(max-min)*100))}%`)};
 sliders.forEach(el=>{paint(el);el.addEventListener('input',()=>paint(el));el.addEventListener('change',()=>paint(el))});
 // Reveal only chapter headings; the visual narrative itself remains scroll controlled.
 const titles=[...document.querySelectorAll('main>section>h2,.opening .scene-copy')];
 const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){entry.target.classList.add('is-visible');observer.unobserve(entry.target)}}),{threshold:.12});
 titles.forEach(el=>{if(!reduced.matches&&!paused()){el.classList.add('premium-reveal-ready');observer.observe(el)}});
 const settle=()=>{if(reduced.matches||paused())titles.forEach(el=>el.classList.add('is-visible'))};
 reduced.addEventListener('change',settle);
 new MutationObserver(settle).observe(document.body,{attributes:true,attributeFilter:['class']});
 // Changes made through existing shortcut buttons also update the slider paint.
 document.addEventListener('click',()=>requestAnimationFrame(()=>sliders.forEach(paint)));
})();
