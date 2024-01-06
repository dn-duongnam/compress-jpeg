<!-- Improved compatibility of back to top link: See: https://github.com/dn-duongnam/compress-jpeg/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the compress-jpeg. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/dn-duongnam/compress-jpeg">
    <img src="uploads/lenna.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Compress-Jpeg</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/dn-duongnam/compress-jpeg"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/dn-duongnam/compress-jpeg">View Demo</a>
    ·
    <a href="https://github.com/dn-duongnam/compress-jpeg/issues">Report Bug</a>
    ·
    <a href="https://github.com/dn-duongnam/compress-jpeg/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Our project focuses on the implementation of various image compression techniques, with a primary emphasis on the JPEG compression algorithm. The key objectives and features of the project are outlined below:

JPEG Algorithm Visualization:
- Display the step-by-step execution of the JPEG image compression algorithm.
- Highlight the following stages: Color space conversion, DCT coefficients before and after quantization, visual representation of DCT coefficients in 3D, selection of - quantization matrices, and evaluation metrics post-compression.

JPEG Image Compression:
- Implement JPEG image compression for both color and grayscale images.
- Provide the flexibility to choose from a range of quantization matrices (Q10-Q100).

DCT-based Denoising Algorithm:
- Develop a denoising algorithm based on Discrete Cosine Transform (DCT).
- Incorporate threshold selection for color and grayscale images.
- For color images, introduce the use of Gaussian Important Map to accelerate denoising.
- Support various noise types such as salt-and-pepper, speckle, and Poisson noise.
- Allow users to adjust thresholds and sliding steps for observation and analysis.

DCT with Median Filter for Grayscale Image Denoising:
- Implement a DCT-based denoising algorithm combined with a median filter.
- Aim to reduce periodic noise in grayscale images.

Through this project, users will gain insights into the inner workings of the JPEG compression algorithm and explore advanced techniques for image denoising. The flexibility provided in parameter selection allows for experimentation and observation of the impact on image quality and compression efficiency.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

This section have list any major frameworks/libraries used in this project. 

* [![Python][Python.com]][Python-url]
* [![Streamlit][Streamlit.com]][Streamlit-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

### Installation


1. Clone the Git repository:
   Open your terminal and run the following command to clone the repository.
   ```sh
   git clone https://github.com/dn-duongnam/compress-jpeg.git
   ```

2. Navigate to the project directory:
   ```sh
   cd compress-jpeg
   ```

3. Install Python requirements:
   Ensure you have Python and pip installed. Then, run the following command to install the required Python packages.
   ```sh
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   Execute the following command to run the Streamlit app.
   ```sh
   streamlit run app.py
   ```
   Or
   ```sh
   python -m streamlit run app.py
   ```

This will start the development server, and you can access the Streamlit app by opening your browser and navigating to the provided URL (usually http://localhost:8501).


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Welcome to the Image Processing website! Below is a detailed guide to help you make the most of the features on the platform.

1. Step 1: Upload Image
   Access the Image Processing website at Link to Your Website.
   You will see a simple interface with the "Upload Image" button. Click on this button to select and upload the image you want to process.

2. Step 2: Adjust Parameters
   Once the image is successfully uploaded, you will see parameter adjustment options displayed on the interface.
   Options may include choosing a quantization matrix, setting thresholds for the denoising algorithm, and other parameters depending on the specific features of the website.

3. Step 3: Process Image
   After adjusting the parameters to your liking, click on the "Process Image" or a similar button.
   The website will proceed to process the image based on the parameters you have selected.

4. Step 4: Observe Results
   The processed image results will be displayed once the operation is complete.
   You can compare the original and processed images to evaluate the performance of the chosen parameter adjustments.

Note:
- Processing times may vary depending on the size of the image and the complexity of the selected parameters.

Enjoy your experience on our Image Processing website! If you have any questions or feedback, feel free to contact us for assistance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] JPEG Compression Step
- [x] JPEG for color image, gray image
- [x] DCT Denoising using Threshold
- [x] DCT Denoising using Median Filter
- [x] Mean/Gaussian Important Map for denoising
- [ ] Unlimit Height/Width of an image (Now is 512x512)
- [ ] Speedup the program

See the [open issues](https://github.com/dn-duongnam/compress-jpeg/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

My team:

- Lã Đức Nam - ducnamla@gmail.com
- Phạm Hồng Nghĩa - phamhongnghia_t65@hus.edu.vn
- Dương Văn Nam - duongvannam_t65@hus.edu.vn

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Lã Đức Nam - ducnamla@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/dn-duongnam/compress-jpeg.svg?style=for-the-badge
[contributors-url]: https://github.com/dn-duongnam/compress-jpeg/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dn-duongnam/compress-jpeg.svg?style=for-the-badge
[forks-url]: https://github.com/dn-duongnam/compress-jpeg/network/members
[stars-shield]: https://img.shields.io/github/stars/dn-duongnam/compress-jpeg.svg?style=for-the-badge
[stars-url]: https://github.com/dn-duongnam/compress-jpeg/stargazers
[issues-shield]: https://img.shields.io/github/issues/dn-duongnam/compress-jpeg.svg?style=for-the-badge
[issues-url]: https://github.com/dn-duongnam/compress-jpeg/issues
[license-shield]: https://img.shields.io/github/license/dn-duongnam/compress-jpeg.svg?style=for-the-badge
[license-url]: https://github.com/dn-duongnam/compress-jpeg/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/badge/Python-563D7C?style=for-the-badge&logo=Python&logoColor=white
[Python-url]: https://getbootstrap.com
[Streamlit]: https://img.shields.io/badge/streamlit-0769AD?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
