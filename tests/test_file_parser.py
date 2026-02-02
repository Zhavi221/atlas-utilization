"""
Tests for FileParser service.

Integration tests that verify file parsing works correctly.
"""

import pytest
import awkward as ak
from unittest.mock import Mock, patch, MagicMock

from src.services.parsing.file_parser import FileParser


class TestFileParser:
    """Tests for FileParser service."""
    
    def test_get_data_tree_name_with_collection_tree(self):
        """Test finding CollectionTree in ROOT file."""
        keys = ["CollectionTree;1", "metadata;1"]
        tree_names = ["CollectionTree", "Events"]
        
        result = FileParser._get_data_tree_name(keys, tree_names)
        assert result == "CollectionTree"
    
    def test_get_data_tree_name_with_events_tree(self):
        """Test finding Events tree in ROOT file."""
        keys = ["Events;1", "metadata;1"]
        tree_names = ["CollectionTree", "Events"]
        
        result = FileParser._get_data_tree_name(keys, tree_names)
        assert result == "Events"
    
    def test_get_data_tree_name_fallback(self):
        """Test fallback to CollectionTree when no match found."""
        keys = ["SomeOtherTree;1"]
        tree_names = ["CollectionTree", "Events"]
        
        result = FileParser._get_data_tree_name(keys, tree_names)
        assert result == "CollectionTree"
    
    def test_can_calculate_inv_mass_with_all_fields(self):
        """Test that all required fields returns True."""
        fields = ["pt", "eta", "phi", "mass"]
        assert FileParser._can_calculate_inv_mass(fields) is True
    
    def test_can_calculate_inv_mass_missing_field(self):
        """Test that missing required field returns False."""
        fields = ["pt", "eta"]  # Missing phi
        assert FileParser._can_calculate_inv_mass(fields) is False
    
    def test_can_calculate_inv_mass_extra_fields(self):
        """Test that extra fields don't affect calculation."""
        fields = ["pt", "eta", "phi", "mass", "charge", "isolation"]
        assert FileParser._can_calculate_inv_mass(fields) is True
    
    @patch('uproot.open')
    def test_parse_file_handles_missing_file(self, mock_open):
        """Test that parsing gracefully handles missing files."""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        result = FileParser.parse_file(
            file_path="/nonexistent/file.root",
            tree_names=["CollectionTree"],
            release_year="2024r-pp",
            batch_size=1000
        )
        
        assert result is None
    
    @patch('uproot.open')
    def test_parse_file_returns_none_when_no_particles(self, mock_open):
        """Test that parsing returns None when no particles found."""
        # Mock ROOT file with no recognizable particle branches
        mock_file = MagicMock()
        mock_file.keys.return_value = ["CollectionTree;1"]
        mock_tree = MagicMock()
        mock_tree.keys.return_value = ["some_branch", "another_branch"]
        mock_tree.num_entries = 100
        mock_file.__getitem__.return_value = mock_tree
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = False
        
        mock_open.return_value = mock_file
        
        result = FileParser.parse_file(
            file_path="/test/file.root",
            tree_names=["CollectionTree"],
            release_year="2024r-pp",
            batch_size=1000
        )
        
        assert result is None
    
    def test_filter_accessible_branches_removes_inaccessible(self):
        """Test that inaccessible branches are filtered out."""
        # Create a mock tree
        mock_tree = MagicMock()
        
        # Mock arrays method to succeed for some branches, fail for others
        def mock_arrays(branch_path, **kwargs):
            if "accessible" in str(branch_path):
                # Return mock awkward array with the field
                mock_arr = MagicMock()
                mock_arr.fields = [branch_path]
                return mock_arr
            else:
                raise Exception("Branch not accessible")
        
        mock_tree.arrays.side_effect = mock_arrays
        
        obj_branches = {
            "Electrons": {
                "AnalysisElectronsAuxDyn.pt_accessible": "pt",
                "AnalysisElectronsAuxDyn.eta_accessible": "eta",
                "AnalysisElectronsAuxDyn.phi_accessible": "phi",
                "AnalysisElectronsAuxDyn.mass_accessible": "mass",
            },
            "Muons": {
                "AnalysisMuonsAuxDyn.pt_inaccessible": "pt",  # All inaccessible
                "AnalysisMuonsAuxDyn.eta_inaccessible": "eta",
            }
        }
        
        result = FileParser._filter_accessible_branches(mock_tree, obj_branches)
        
        # Should keep only Electrons (all accessible branches)
        assert "Electrons" in result
        assert len(result["Electrons"]) == 4
        assert "AnalysisElectronsAuxDyn.pt_accessible" in result["Electrons"]
        
        # Muons should be filtered out (not enough accessible branches for inv mass)
        assert "Muons" not in result
    
    def test_read_file_in_batches_single_batch(self):
        """Test reading file in a single batch."""
        # Create mock tree
        mock_tree = MagicMock()
        
        # Mock arrays method
        mock_data = ak.Array([
            {"AnalysisElectronsAuxDyn.pt": [10.0, 20.0], "AnalysisElectronsAuxDyn.eta": [1.0, 2.0]}
        ])
        mock_tree.arrays.return_value = mock_data
        
        all_branches = {"AnalysisElectronsAuxDyn.pt", "AnalysisElectronsAuxDyn.eta"}
        obj_branches = {
            "Electrons": {
                "AnalysisElectronsAuxDyn.pt": "pt",
                "AnalysisElectronsAuxDyn.eta": "eta",
            }
        }
        
        result = FileParser._read_file_in_batches(
            tree=mock_tree,
            all_branches=all_branches,
            obj_branches=obj_branches,
            n_entries=100,
            batch_size=1000  # Larger than n_entries, so single batch
        )
        
        # Should have Electrons object
        assert "Electrons" in result
        assert isinstance(result["Electrons"], ak.Array)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
