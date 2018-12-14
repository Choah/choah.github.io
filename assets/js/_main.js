/* ========================================================================== 
2     jQuery plugin settings and other scripts 
3     ========================================================================== */ 
4  
 
5  $(document).ready(function() { 
6    // Sticky footer 
7    var bumpIt = function() { 
8        $("body").css("margin-bottom", $(".page__footer").outerHeight(true)); 
9      }, 
10      didResize = false; 
11  
 
12    bumpIt(); 
13  
 
14    $(window).resize(function() { 
15      didResize = true; 
16    }); 
17    setInterval(function() { 
18      if (didResize) { 
19        didResize = false; 
20        bumpIt(); 
21      } 
22    }, 250); 
23  
 
24    // FitVids init 
25    $("#main").fitVids(); 
26  
 
27    // Sticky sidebar 
28    var stickySideBar = function() { 
29      var show = 
30        $(".author__urls-wrapper button").length === 0 
31          ? $(window).width() > 1024 // width should match $large Sass variable 
32          : !$(".author__urls-wrapper button").is(":visible"); 
33      if (show) { 
34        // fix 
35        $(".sidebar").addClass("sticky"); 
36      } else { 
37        // unfix 
38        $(".sidebar").removeClass("sticky"); 
39      } 
40    }; 
41  
 
42    stickySideBar(); 
43  
 
44    $(window).resize(function() { 
45      stickySideBar(); 
46    }); 
47  
 
48    // Follow menu drop down 
49    $(".author__urls-wrapper button").on("click", function() { 
50      $(".author__urls").toggleClass("is--visible"); 
51      $(".author__urls-wrapper button").toggleClass("open"); 
52    }); 
53  
 
54    // Search toggle 
55    $(".search__toggle").on("click", function() { 
56      $(".search-content").toggleClass("is--visible"); 
57      $(".initial-content").toggleClass("is--hidden"); 
58      // set focus on input 
59      setTimeout(function() { 
60        $(".search-content input").focus(); 
61      }, 400); 
62    }); 
63  
 
64    // init smooth scroll 
65    $("a").smoothScroll({ offset: -20 }); 
66  
 
67    // add lightbox class to all image links 
68    $( 
69      "a[href$='.jpg'],a[href$='.jpeg'],a[href$='.JPG'],a[href$='.png'],a[href$='.gif']" 
70    ).addClass("image-popup"); 
71  
 
72    // Magnific-Popup options 
73    $(".image-popup").magnificPopup({ 
74      // disableOn: function() { 
75      //   if( $(window).width() < 500 ) { 
76      //     return false; 
77      //   } 
78      //   return true; 
79      // }, 
80      type: "image", 
81      tLoading: "Loading image #%curr%...", 
82      gallery: { 
83        enabled: true, 
84        navigateByImgClick: true, 
85        preload: [0, 1] // Will preload 0 - before current, and 1 after the current image 
86      }, 
87      image: { 
88        tError: '<a href="%url%">Image #%curr%</a> could not be loaded.' 
89      }, 
90      removalDelay: 500, // Delay in milliseconds before popup is removed 
91      // Class that is added to body when popup is open. 
92      // make it unique to apply your CSS animations just to this exact popup 
93      mainClass: "mfp-zoom-in", 
94      callbacks: { 
95        beforeOpen: function() { 
96          // just a hack that adds mfp-anim class to markup 
97          this.st.image.markup = this.st.image.markup.replace( 
98            "mfp-figure", 
99            "mfp-figure mfp-with-anim" 
100          ); 
101        } 
102      }, 
103      closeOnContentClick: true, 
104      midClick: true // allow opening popup on middle mouse click. Always set it to true if you don't provide alternative source. 
105    }); 
106  }); 
