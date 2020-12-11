'use strict';
const http = require('http');
var assert = require('assert');
const express= require('express');
const app = express();
const mustache = require('mustache');
const filesystem = require('fs');
const url = require('url');

//const hostname = '127.0.0.1';
const port = Number(process.argv[2]);

const hbase = require('hbase')
var hclient = hbase({ host: process.argv[3], port: Number(process.argv[4])})

function rowToMap(row) {
	var stats = {}
	row.forEach(function (item) {
		stats[item['column']] = Number(item['$'])
	});
	return stats;
}

hclient.table('kuanhc96_project_hbase').row('100000026.jpg').get((error, value) => {
	console.info(rowToMap(value))
	//console.info(value)
})

app.use(express.static('public'));
app.get('/delays.html',function (req, res) {

	const rgbDictionary = {
		"red": [220, 20, 60],
		"maroon": [128, 0, 0],
		"tan": [210, 180, 140],
		"orange": [255, 165, 0],
		"gold": [255, 215, 0],
		"green": [0, 128, 0],
		"lime": [50, 205, 50],
		"blue": [0, 0, 255],
		"navy": [0, 0, 128],
		"yellow": [254, 254, 34],
		"cyan": [0, 255, 255],
		"magenta": [255, 0, 255],
		"pink": [255, 192, 203],
		"purple": [128, 0, 128],
		"indigo": [75, 0, 130],
		"teal": [0, 128, 128],
		"olive": [128, 128, 0],
		"gray": [128, 128, 128],
		"brown": [160, 82, 45],
		"white": [255, 255, 255],
		"black": [0, 0, 0]
	}

	var person=req.query['id'] + "0000";
	var prefix;
	if (req.query['frame'] < 10) {
		prefix = "000"
	} else if (req.query['frame'] < 100) {
		prefix = "00"
	} else if (req.query['frame'] < 1000) {
		prefix = "0"
	}
	person = person + prefix + req.query['frame'] + ".jpg"



    console.log(person);
	hclient.table('kuanhc96_project_hbase').row(person).get(function (err, cells) {
		if (cells === null) {
			var template = filesystem.readFileSync("result.mustache").toString();
			var html = mustache.render(template,  {
				id : "No Data",
				frame_id : "No Data",
				xl : "No Data",
				yl : "No Data",
				xr : "No Data",
				yr : "No Data",
				red : "No Data",
				blue : "No Data",
				green : "No Data"
			});
			res.send(html);
		} else {
			const personInfo = rowToMap(cells);

			function process(frame_id) {
				if (frame_id > 5289) {
					throw "Illegal argument"
				}
				console.log(frame_id + "/5289")
				return frame_id + "/5289"
			}

			function getColorLabel(r, g, b) {
				var shortest = Infinity;
				var currentColor = "";
				for (var key in rgbDictionary) {
					const val = rgbDictionary[key]
					const dist = Math.pow((r-val[0]), 2) + Math.pow((g-val[1]), 2) + Math.pow((b-val[2]), 2);
					if (dist < shortest) {
						shortest = dist;
						currentColor = key;
					}
				}
				return currentColor;
			}

			var template = filesystem.readFileSync("result.mustache").toString();
			var html = mustache.render(template,  {
				id : req.query['id'],
				frame_id : process(req.query['frame']),
				xl : personInfo["status:xl"],
				yl : personInfo["status:yl"],
				xr : personInfo["status:xr"],
				yr : personInfo["status:yr"],
				red : personInfo["status:red"],
				blue : personInfo["status:blue"],
				green : personInfo["status:green"],
				color : getColorLabel(personInfo["status:red"], personInfo["status:green"], personInfo["status:blue"]),
				image : "/frames/0000" + prefix + req.query['frame'] + ".jpg",
				width : personInfo["status:xr"] - personInfo["status:xl"],
				height: personInfo["status:yr"] - personInfo["status:yl"]

			});


			//ctx.drawImage(image,

			//	)

			res.send(html);
		}

	});
});
	
app.listen(port);


