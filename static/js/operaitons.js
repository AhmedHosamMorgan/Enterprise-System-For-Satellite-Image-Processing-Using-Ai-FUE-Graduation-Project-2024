// upload img

var upload_img_btn = document.getElementById('upload_img_btn');
var imgFile = document.getElementById('imgFile');

var form = new FormData()

upload_img_btn.addEventListener('click',function(){
    imgFile.click()
})

let imageType ;

imgFile.addEventListener('change', (e) => {
    var img = e.target.files[0];
    imageType = img.name;

    var imgSrc = URL.createObjectURL(img);

    form.append('img', img);

    var image_el = `
        <div class="image">
            <img src="${imgSrc}" alt="">
            <p>${img.name}</p>
        </div>
    `;

    document.querySelector('.col').innerHTML += image_el;

    function addOrUpdateImageOverlay(imgName, imageUrl, southWestLatitude, southWestLongitude, northEastLatitude, northEastLongitude, opacity) {
        var bounds = [[southWestLatitude, southWestLongitude], [northEastLatitude, northEastLongitude]];
    
        if (!map.hasLayer(imageOverlay)) {
            var imageOverlay = L.imageOverlay(imageUrl, bounds, { opacity: opacity }).addTo(map);
        } else {
            imageOverlay.setUrl(imageUrl).setBounds(bounds).setOpacity(opacity);
        }
    }
    
    if (img.name === 'Delta.png') {
        addOrUpdateImageOverlay('Delta.png', '/static/Dataset/Delta.png',
            ((-4383.5 + 2229.5) / 10000) + 30.4,
            ((3856.5 - 5930.5) / 10000) + 32.2 - (5 / (111.32 * Math.cos((30.4 + 30.8) / 2 * Math.PI / 180))),
            ((-2229.5 + 4383.5) / 10000) + 30.8,
            ((5930.5 - 3856.5) / 10000) + 32.7 - (5 / (111.32 * Math.cos((30.4 + 30.8) / 2 * Math.PI / 180))),
            0.8);
    } 
    
    else if (img.name === 'NoisedImage.jpg') {
        addOrUpdateImageOverlay('NoisedImage.jpg', '/static/Dataset/NoisedImage.jpg',
            ((-4383.5 + 2229.5) / 10000) + 30.4,
            ((3856.5 - 5930.5) / 10000) + 32.2 - (5 / (111.32 * Math.cos((30.4 + 30.8) / 2 * Math.PI / 180))),
            ((-2229.5 + 4383.5) / 10000) + 30.8,
            ((5930.5 - 3856.5) / 10000) + 32.7 - (5 / (111.32 * Math.cos((30.4 + 30.8) / 2 * Math.PI / 180))),
            0.02);
    } 
    
    else if (img.name === 'Cairo.png' || img.name === 'Cairo.jpg') {
        addOrUpdateImageOverlay('Cairo.png', '/static/Dataset/Cairo.png',
        29.80+0.10, 30.90+0.10, 30.20+0.10, 31.45+0.10, 0.8);
    }

    else if (img.name === 'Fayoum.png' || img.name === 'Fayoum.jpg') {
        addOrUpdateImageOverlay('Fayoum.png', '/static/Dataset/Fayoum.png',
        29.10-0.06, 30.15+0.18, 29.70-0.06, 31.00+0.18, 0.8);
    }

    else if (img.name === 'Qena.png' || img.name === 'Qena.jpg') {
        addOrUpdateImageOverlay('Qena.png', '/static/Dataset/Qena.png',
        25.50-0.15, 32.00-0.18, 26.90-0.15, 33.30-0.18, 0.8);
    }

    else if (img.name === 'Sharm.png' || img.name === 'Sharm.jpg') {
        addOrUpdateImageOverlay('Sharm.png', '/static/Dataset/Sharm.png',
        27.30+0.14, 33.80-0.25, 28.50+0.14, 35.25-0.25, 0.8);
    }

    else if (img.name === 'PortSaid.png'|| img.name === 'PortSaid.jpg') {
        addOrUpdateImageOverlay('PortSaid.png', '/static/Dataset/PortSaid.png',
        30.80-0.03, 32.00-0.01, 31.51-0.03, 32.70-0.01, 0.8);
    }

    else if (img.name === 'NasserLake.png' || img.name === 'NasserLake.jpg' ) {
        addOrUpdateImageOverlay('NasserLake.png', '/static/Dataset/NasserLake.png',
        23.00-0.40, 31.00+0.90, 24.00-0.40, 33.00+0.90, 0.8);
    }

    else {

        alert("This Image Does Not Contain the Geographical Coordinates Necessary to Add the Image to The Map in Its Correct Place. Accordingly, We Will Do the Processing in The Background and Download the Image to Your Computer Automatically.");
        if (map.hasLayer(imageOverlay)) {
            map.removeLayer(imageOverlay);
        }
    }
});



var methodsEles = document.querySelectorAll('.title.method');
var DisCol = document.querySelector('.col.dis');




methodsEles.forEach( method => {

    method.addEventListener('click',function(){
        
        methodsEles.forEach(m=>m.classList.remove('active'))

        method.classList.add('active')

        DisCol.classList.remove('dis');

        var method_type = method.querySelector('p').textContent;

        DisCol.querySelectorAll('.title.function').forEach(i=>i.remove())
        
        if ( method_type == 'Histogram Stretching'){
            
            Methods.HistogramStreaming.forEach( item =>{
                var name = item[0];
                var ajax_fun_name = item[1];
                DisCol.innerHTML += `
                <div class="title function" >
                    <p id="${ajax_fun_name}" >${name}</p>
                </div>
            
            `
            })
            
        }
    
        if ( method_type == 'Image Filtration'){
            Methods.ImageFilteration.forEach( item =>{
                var name = item[0];
                var ajax_fun_name = item[1];
                DisCol.innerHTML += `
                <div class="title function" >
                    <p id="${ajax_fun_name}">${name}</p>
                </div>
            
            `
            })
        }
    
        if ( method_type == 'Image Transformation'){
            Methods.ImageTransformation.forEach( item =>{
                var name = item[0];
                var ajax_fun_name = item[1];
                DisCol.innerHTML += `
                <div class="title function" >
                    <p id="${ajax_fun_name}">${name}</p>
                </div>
            
            `
            })
        }
        
        if ( method_type == 'Super Resolution'){
            Methods.ImageRes.forEach( item =>{
                var name = item[0];
                var ajax_fun_name = item[1];
                DisCol.innerHTML += `
                <div class="title function" >
                    <p id="${ajax_fun_name}">${name}</p>
                </div>
            
            `
            })
        }

    
        var functions = DisCol.querySelectorAll('.title.function');

        functions.forEach( func=>{

            func.addEventListener('click',()=>{
                functions.forEach(i=>i.classList.remove('active'));
                func.classList.add('active')
                ajax_url = func.querySelector('p').id ;
            })
            
        })



    })

    
})



var RunProcess = document.querySelector('.runBtn');

RunProcess.addEventListener('click',()=>{

    if (ajax_url){
        RunProcess.classList.toggle('load')
        RunProcess.textContent = 'Loading...'
        RunProcess.disabled = true

        console.log(ajax_url)
        $.ajax({
            url:ajax_url,
            type:"POST",
            data:form,
            contentType:false,
            processData:false,
            cache:false,
            success: function(res) {
                var imageUrl = res; 
                


                
    function addOrUpdateImageOverlay(imgName, imageUrl, southWestLatitude, southWestLongitude, northEastLatitude, northEastLongitude, opacity) {
        var bounds = [[southWestLatitude, southWestLongitude], [northEastLatitude, northEastLongitude]];
    
        if (!map.hasLayer(imageOverlay)) {
            var imageOverlay = L.imageOverlay(imageUrl, bounds, { opacity: opacity }).addTo(map);
        } else {
            imageOverlay.setUrl(imageUrl).setBounds(bounds).setOpacity(opacity);
        }
    }
    
    if (imageType === 'Delta.png' || imageType === 'Delta.jpg') {
        addOrUpdateImageOverlay('Delta.png', imageUrl,
            ((-4383.5 + 2229.5) / 10000) + 30.4,
            ((3856.5 - 5930.5) / 10000) + 32.2 - (5 / (111.32 * Math.cos((30.4 + 30.8) / 2 * Math.PI / 180))),
            ((-2229.5 + 4383.5) / 10000) + 30.8,
            ((5930.5 - 3856.5) / 10000) + 32.7 - (5 / (111.32 * Math.cos((30.4 + 30.8) / 2 * Math.PI / 180))),
            0.95);

            downloadImage(imageUrl, 'processed_image.png'); 
    } 
    
    else if (imageType === 'NoisedImage.jpg' || imageType === 'NoisedImage.jpg') {

        addOrUpdateImageOverlay('NoisedImage.jpg',imageUrl,
            ((-4383.5 + 2229.5) / 10000) + 30.4,
            ((3856.5 - 5930.5) / 10000) + 32.2 - (5 / (111.32 * Math.cos((30.4 + 30.8) / 2 * Math.PI / 180))),
            ((-2229.5 + 4383.5) / 10000) + 30.8,
            ((5930.5 - 3856.5) / 10000) + 32.7 - (5 / (111.32 * Math.cos((30.4 + 30.8) / 2 * Math.PI / 180))),
            0.95);

            downloadImage(imageUrl, 'processed_image.png'); 

    } 
    
    else if (imageType === 'Cairo.png' || imageType === 'Cairo.jpg') {

        addOrUpdateImageOverlay('Cairo.png',imageUrl,
        29.80+0.10, 30.90+0.10, 30.20+0.10, 31.45+0.10, 0.95);
        downloadImage(imageUrl, 'processed_image.png'); 
    }

    else if (imageType === 'Fayoum.png' || imageType === 'Fayoum.jpg') {

        addOrUpdateImageOverlay('Fayoum.png', imageUrl,
        29.10-0.06, 30.15+0.18, 29.70-0.06, 31.00+0.18, 0.95);
        downloadImage(imageUrl, 'processed_image.png');

    }

    else if (imageType === 'Qena.png' || imageType === 'Qena.jpg') {

        addOrUpdateImageOverlay('Qena.png', imageUrl,
        25.50-0.15, 32.00-0.18, 26.90-0.15, 33.30-0.18, 0.95);
        downloadImage(imageUrl, 'processed_image.png'); 

    }

    else if (imageType === 'Sharm.png' || imageType === 'Sharm.jpg') {

        addOrUpdateImageOverlay('Sharm.png', imageUrl,
        27.30+0.14, 33.80-0.25, 28.50+0.14, 35.25-0.25, 0.95);
        downloadImage(imageUrl, 'processed_image.png'); 

    }

    else if (imageType === 'PortSaid.png' || imageType === 'PortSaid.jpg') {

        addOrUpdateImageOverlay('PortSaid.png',imageUrl,
        30.80-0.03, 32.00-0.01, 31.51-0.03, 32.70-0.01, 0.95);
        downloadImage(imageUrl, 'processed_image.png'); 

    }

    else if (imageType === 'NasserLake.png' || imageType === 'NasserLake.jpg') {

        addOrUpdateImageOverlay('NasserLake.png', imageUrl,
        23.00-0.40, 31.00+0.90, 24.00-0.40, 33.00+0.90, 0.95);
        downloadImage(imageUrl, 'processed_image.png'); 

    }

    else {

        downloadImage(imageUrl, 'processed_image.png'); 

        if (map.hasLayer(imageOverlay)) {
            map.removeLayer(imageOverlay);
        }
    }
      
                RunProcess.classList.toggle('load');
                RunProcess.textContent = 'Run';
                RunProcess.disabled = false;
            }
        })

        
    }else{
        alert('Please Chooose image Function')
    }
    
})


function downloadImage(url, filename) {
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}