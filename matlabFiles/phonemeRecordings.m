


frameLength = 1/30;

time = 0;

startTimeOfPhonemes = table2array(LipSpeaker(:,1))/10^7;
endTime = table2array(LipSpeaker(:,2))/10^7;
phonemes = table2array(LipSpeaker(:,3));

notNans = find(~isnan(startTimeOfPhonemes));

newPhonemeList = phonemes(notNans,:);
startTimeOfPhonemes = startTimeOfPhonemes(notNans);
endTime = endTime(notNans);

framePhonemeLabels = [];

i = 1;

indexOfVideoEndTime = find(~startTimeOfPhonemes)-1;
indexOfVideoEndTime = [indexOfVideoEndTime(2:end); length(endTime)];
indexOfVideoStartTime = find(~startTimeOfPhonemes);

numOfVids = length(find(~startTimeOfPhonemes));


for j = 1:numOfVids
    
    time = 0;
    framePhonemeLabels = [];
    
    startIndex = indexOfVideoStartTime(j);
    endIndex = indexOfVideoEndTime(j);
    
    inputPhonemes = newPhonemeList(startIndex:endIndex);
    
    i=1;
    
    while(time<endTime(indexOfVideoEndTime(j)))
        
        time = time + frameLength;
        if ( time > startTimeOfPhonemes(i) && time < endTime(i))
            if(i <= length(inputPhonemes))
                framePhonemeLabels = [framePhonemeLabels inputPhonemes(i)];
            end
        else
            i = i+1;
            if(i <= length(inputPhonemes))
                framePhonemeLabels = [framePhonemeLabels inputPhonemes(i)];
            end
        end
        
    end

    name = char(strcat('Recording',int2str(j)));
    phonemeSeqCells.(name) = framePhonemeLabels;
    
    
end


