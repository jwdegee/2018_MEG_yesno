subjects = {'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10',...
    'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19',...
    'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30'};

fs_subject_dir = '/Users/nwilming/lisa_cluster/subjects';

for ii = 1:length(subjects)
    make_mesh(subjects{ii}, fs_subject_dir);
end