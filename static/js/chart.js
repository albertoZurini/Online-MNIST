google.charts.load("current", {packages:["corechart"]});

var probArr = [];

function drawChart() {
  var data = google.visualization.arrayToDataTable(probArr);

  var options = {
    title: 'CNN result',
    legend: { position: 'none' },
  };

  var chart = new google.visualization.BarChart(document.getElementById('chart_div'));
  chart.draw(data, options);
}
