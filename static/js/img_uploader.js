const dropArea = document.getElementById("drop-area");
const inputFile = document.getElementById("input-file");

inputFile.addEventListener("change", uploadImage);

function uploadImage() {
    
    let imgLink = URL.createObjectURL(inputFile.files[0]);
    
    let img = inputFile.files[0];
    var form = new FormData()
            
    

    form.append('img',img)
    var url = document.getElementById('ajax_url').value ;

    $.ajax({
        url:url,
        // url:"/blur-img/",
        type:"POST",
        data:form,
        contentType:false,
        processData:false,
        cache:false,
        success:function(res){

            var href = "/" + res;

            var url = document.createElement('a');
            url.setAttribute('href',href)
            url.setAttribute('download',true)

            url.click()
            
        }
    })


    // window.open(imgLink, "_blank");
}

dropArea.addEventListener("dragover", function (e) {
    e.preventDefault();
});

dropArea.addEventListener("drop", function (e) {
    e.preventDefault();
    inputFile.files = e.dataTransfer.files;
    uploadImage();
});
