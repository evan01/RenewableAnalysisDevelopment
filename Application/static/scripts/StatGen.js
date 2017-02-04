function sendData() {
    console.log("Sup evan")
}

$scope.uploadFile = function(element){
	$scope.$apply(function($scope){
		$scope.files = element.files;
	});
}

$scope.addFile = function(){
	UploadService.uploadfile($scope.files,
		function(msg) //success
		{
			console.log('uploaded');
		},
		function(msg) //error
		{
			console.log('error');
		});
}

uploadfile: function (files,success,error){
	
	var url = 'file:///C:/Users/Julia/Documents/RenewableAnalysisDevelopment/Application/static/index.html';

	for(var i = 0; i < files.length; i++)
	{
		var fd = new FormData();

		fd.append("file", files[i]);
		$http.post(url, fd, {

			withCredentials : false,

			headers : {
				'Content-Type' : undefined
			},
			transformRequest : angular.identity
		})
		.success(function(data)
		{
			console.log(data);
		})
		.error(function(data)
		{
			console.log(data);
		});
	}
}