% load HDR
hdr = hdrread('../Part1/Results/hdr_image.hdr');

% tone-map
keys=[0.02, 0.07, 0.10];
burns=[0.1, 0.5, 1];

fignum = 3;

for key=keys

    imwrite( ...
        reinhard_tonemapping(hdr, key, 1, fignum), ...
        "Results/tonemapped_reinhard_key_" + key + ".png" ...
        )

end

for burn=burns

    imwrite( ...
        reinhard_tonemapping(hdr, keys(2), burn, fignum+1), ...
        "Results/tonemapped_reinhard_burn_" + burn + ".png" ...
        )

end

simple_hdr = scale_and_gamma(hdr ./ (1 + hdr));

clf(figure(fignum+2))
figure(fignum+2)
imagesc(simple_hdr)
axis image;
    
imwrite(simple_hdr, "Results/simple_hdr.png")

clear fignum key burn;
