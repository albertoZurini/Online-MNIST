function showLoading(){
  $('#loader').show();
}
function hideLoading(){
  $('#loader').hide();
}

$('#predict-btn').click(function(){
  let canvas = document.getElementById('sheet');
  var url = canvas.toDataURL('image/png');

  showLoading();

  $.ajax({
    url: '/predict',
    method: 'post',
    data: {img: url},
    success: function(data){
      $('#pred').text('You have written a'+([1, 8].indexOf(data.val) !== -1 ? 'n': '')+' '+data.val);

      probArr = [['Number', 'Probability']];
      for(var index in data.stats){
        probArr.push([index, data.stats[index]]);
      }
      
      drawChart();
      hideLoading();
    },
    error: function(err){
      console.error('Error: ', err);

      $('#pred').text('Network error');

      hideLoading();
    }
  });
});